import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class DisNet(object):
    def __init__(self,bZoomCamera,scale=1):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.detect_classes_path = 'model_data/detect_classes.txt'
        self.distance_model_path = "model_3.keras"
        self.score = 0.3
        self.iou = 0.5
        self.zoom_in_factor = scale
        self.ZoomCamera=bZoomCamera
        self.class_names = self._get_class()
        self.detect_classes = self._get_detect_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (960, 960)  # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_detect_class(self):
        classes_path = os.path.expanduser(self.detect_classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        self.set_class_size()
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        self.distance_model = load_model(self.distance_model_path, compile=False)

        hsv_tuples = [(float(x) / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = time.time()

        if not self.ZoomCamera:
            if self.zoom_in_factor != 1:
                image = self.zoom_in_image(image)

        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(1.5e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 900

        for i, c in reversed(list(enumerate(out_classes))):
            if self.class_names[c] in self.detect_classes:
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                distance_input = self.load_dist_input(box, predicted_class, image.width, image.height)
                distance = self.distance_model.predict(np.array([distance_input]).reshape(-1, 6)) * self.zoom_in_factor

                label = '{} {:.2f} \n Distance {:.2f}'.format(predicted_class, float(np.squeeze(score)), float(np.squeeze(distance)))
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        end = time.time()
        print("FPS: {:.2f}".format(1.0/(end - start)))
        return image

    def zoom_in_image(self,image):
        image = np.asarray(image)
        image_center = np.divide(image.shape[:2], 2)
        resize_height = np.round(image.shape[0] * float(1.0/self.zoom_in_factor))
        resize_width = np.round(image.shape[1] * float(1.0/self.zoom_in_factor))

        width_begin = np.int(image_center[1] - resize_width / 2)
        width_end = np.int(image_center[1] + resize_width / 2)
        height_begin = np.int(image_center[0] - resize_height / 2)
        height_end = np.int(image_center[0] + resize_height / 2)
        image_crop = image[height_begin:height_end,width_begin:width_end]
        image_zoom = cv2.resize(image_crop, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(image_zoom)

    def load_dist_input(self, predict_box, predict_class, img_width, img_height):
        top, left, bottom, right = predict_box
        width = float(right - left) / img_width
        height = float(bottom - top) / img_height
        diagonal = np.sqrt(np.square(width) + np.square(height))
        class_h, class_w, class_d = np.array(self.set_class_size[predict_class], dtype=np.float32)
        dist_input = [1 / width, 1 / height, 1 / diagonal, class_h, class_w, class_d]
        return np.array(dist_input)

    def set_class_size(self):
        classes = ['person', 'bus', 'truck', 'car', 'bicycle', 'motorbike', 'cat', 'dog', 'horse', 'sheep', 'cow']
        class_shape = [[175, 55, 30], [300, 250, 1200], [300, 250, 1200], [160, 180, 400], [110, 50, 180],
                       [110, 50, 180], [40, 20, 50], [50, 30, 60], [180, 60, 200], [130, 60, 150], [170, 70, 200]]
        self.set_class_size = dict(zip(classes, class_shape))
        print("Load Class size!")

    def close_session(self):
        self.sess.close()