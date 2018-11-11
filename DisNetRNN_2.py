import os
import cv2
import random
import colorsys
import numpy as np

import keras.backend as k
from timeit import time

from keras.models import Sequential, Model, load_model
from keras.layers import Input, LSTM, Dense, Reshape, Dropout,GRU
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.externals import joblib

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image
from PIL import Image, ImageFont, ImageDraw


class DisNet_RNN(object):
    def __init__(self, bZoomCamera=False, scale=1):
        self.img_width = 2592
        self.img_height = 1944
        self.counter_tracker_len = np.array([])
        self.counter_untrack_len = np.array([])
        self.tracker_last = np.array([])
        self.tracker_new = np.array([])
        self.detection_new = np.array([])
        self.tracker_color = np.array([], dtype=np.int32)
        self.prediction_result = np.array([])
        self.prediction_tracker_index = np.array([])
        self.class_size_dict = self.set_class_size()

        self.yolo_model_path = 'model_data/yolo.h5'
        self.DisNet_weights_path = "DisNet_RNN_Model/DisNet_RNN_vg_ss_weights.h5"
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.detect_classes_path = 'model_data/detect_classes.txt'

        self.yolo_score = 0.3
        self.yolo_iou = 0.5
        self.model_image_size = (960, 960)  # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)

        self.sess = k.get_session()
        self.class_names = self._get_class()
        self.detect_classes = self._get_detect_class()
        self.yolo_class_colors = self._yolo_class_color()
        self.anchors = self._get_anchors()
        self.scaler = self._get_scaler()
        self.boxes, self.scores, self.classes = self.load_yolov3_model()

        self.zoom_in_factor = scale
        self.ZoomCamera = bZoomCamera

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

    def _get_scaler(self):
        scaler_path = os.path.expanduser("DisNet_RNN_Model/model_vg_stand_scaler.sav")
        if os.path.isfile(scaler_path):
            scaler = joblib.load(scaler_path)
            return scaler
        else:
            print("Data Scaler file is not found.")

    def _yolo_class_color(self):
        hsv_tuples = [(float(x) / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                          colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)
        return colors

    def _tracker_color(self, length):
        for i in range(length):
            new_color = list(colorsys.hsv_to_rgb(np.random.random_sample(), 1., 1.))
            new_color = [int(new_color[0] * 255), int(new_color[1] * 255), int(new_color[2] * 255)]
            while new_color in self.tracker_color.tolist():
                new_color = list(colorsys.hsv_to_rgb(np.random.random_sample(), 1., 1.))
                new_color = [int(new_color[0] * 255), int(new_color[1] * 255), int(new_color[2] * 255)]
            if not len(self.tracker_color):
                self.tracker_color = np.append(self.tracker_color, np.array(new_color, dtype=np.int32))
            else:
                self.tracker_color = np.append(self.tracker_color, np.array([new_color]), axis=0)
            self.tracker_color = self.tracker_color.reshape(-1, 3)

    def set_class_size(self):
        classes = ['person', 'bus', 'truck', 'car', 'bicycle', 'motorbike']
        class_shape = [[1.75, 0.55, 0.30], [3.00, 2.50, 12.00], [3.00, 2.50, 12.00], [1.60, 1.80, 4.00],
                       [1.10, 0.50, 1.80],
                       [1.10, 0.50, 1.80]]
        print("Load Class size!")
        return dict(zip(classes, class_shape))

    def data_generator(self, predict_boxes, predict_classes):
        detection_cache = []
        if len(predict_boxes) is 0:
            return np.array([])
        else:
            for i in range(len(predict_boxes)):
                top, left, bottom, right = predict_boxes[i]
                top_pre = (float(top) / self.img_height)
                left_pre = (float(left) / self.img_width)
                width = float(right - left) / self.img_width
                height = float(bottom - top) / self.img_height
                diagonal = np.sqrt(np.square(width) + np.square(height))
                predict_class = predict_classes[i]
                class_h, class_w, class_d = np.array(self.class_size_dict[predict_class], dtype=np.float32) * 100
                current_data = [class_h, class_w, class_d, 1.0 / width, 1.0 / height, 1.0 / diagonal,top_pre, left_pre]
                detection_cache.append(current_data)
            return np.array(detection_cache)

    def box_convert(self, detection_output=np.array([])):
        if len(detection_output) is 0:
            return np.array([])
        else:
            detection_box = []
            for output in detection_output:
                if np.any(output):
                    width = 1.0 / output[3]
                    height = 1.0 / output[4]
                    x1 = output[6]
                    y1 = output[7]
                    x2 = x1 + height
                    y2 = y1 + width
                    detection_box.append([x1, y1, x2, y2])
                else:
                    detection_box.append([0, 0, 0, 0])
            return np.array(detection_box)

    ## convert rnn output to input format
    def Output_convert(self, rnn_out):
        convert_output = np.zeros_like(rnn_out)
        convert_output[:, :3] = 100.0 / rnn_out[:, :3]
        convert_output[:, 3:] = rnn_out[:, 3:]
        return convert_output

    def iou(self, bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.01):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)
        matched_indices = linear_assignment(-iou_matrix)
        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update_tracker(self, predict_boxes, predict_class_names, predict_classes, detection_index):
        # Generate Tracker sequence for RNN input

        # Detection result of current frame
        self.detection_new = self.data_generator(predict_boxes, predict_class_names)

        # if no detection and no tracker, Do nothing just return function
        if len(self.tracker_last) is 0:
            if len(self.detection_new) is 0:
                return
            else:
                # if Detection result exist, but no tracker. Then initialize new tracker the same number as detector
                self.tracker_new = np.zeros((len(self.detection_new), 3, 8))
                self.tracker_new[:, -1, :] = self.detection_new

                self.tracker_detect_index = np.array(detection_index)

                ##############
                self.tracker_class = np.array(predict_classes)

        else:

            # if no detection, but tracker_last exists. Then use prediction result of this frame to update tracker new
            # And increase tracker without detection counter
            if len(self.detection_new) is 0:
                self.tracker_new = np.zeros_like(self.tracker_last)
                self.tracker_new[:, :-1, :] = self.tracker_last[:, 1:, :]

            else:

                # if both detection result and tracker_last exist, match tracker with detector and update tracker_new with detection result

                # generate detection boxes and tracker boxes of last frame
                self.detection_box = self.box_convert(self.detection_new)
                self.tracker_box = self.box_convert(self.tracker_last[:, -1, :])

                # associate detection_result with tracker
                matches, unmatched_detections, unmatched_trackers = self.associate_detections_to_trackers(
                    self.detection_box,
                    self.tracker_box)
                self.matches = matches
                self.unmatches = unmatched_detections
                self.unmatched_trackers = unmatched_trackers

                # update tracker with corresponding detection results
                self.tracker_new = np.zeros((len(matches) + len(unmatched_trackers) + len(unmatched_detections), 3, 8))
                self.tracker_new[:len(self.tracker_last), :-1, :] = self.tracker_last[:, 1:, :]
                self.tracker_new[matches[:, 1], -1, :] = self.detection_new[matches[:, 0]]

                self.tracker_detect_index = np.zeros((len(matches) + len(unmatched_trackers) + len(unmatched_detections)))
                self.tracker_detect_index[matches[:, 1]] = detection_index[matches[:, 0]]

                # create new trackers for unmatched_detections result
                if len(unmatched_detections) is not 0:
                    self.tracker_new[len(self.tracker_last):, -1, :] = self.detection_new[unmatched_detections]

                    self.tracker_detect_index[(len(matches) + len(unmatched_trackers)):] = detection_index[unmatched_detections]

                    #########
                    self.tracker_class_new = np.zeros(
                        len(matches) + len(unmatched_trackers) + len(unmatched_detections), np.int32)
                    self.tracker_class_new[:len(self.tracker_class)] = self.tracker_class
                    self.tracker_class_new[len(self.tracker_class):] = np.array(predict_classes)[
                        np.squeeze(unmatched_detections)]
                    self.tracker_class = self.tracker_class_new

        # delete trackers without detection for a long time
        self.delete_tracker_indexs = np.squeeze(np.argwhere(np.count_nonzero(self.tracker_new[:,-1],axis=1)==0))

        # update trackers and helpful counter
        self.tracker_new = np.delete(self.tracker_new, self.delete_tracker_indexs, axis=0)


        ####################
        self.tracker_class = np.delete(self.tracker_class, self.delete_tracker_indexs)
        self.tracker_detect_index = np.delete(self.tracker_detect_index,self.delete_tracker_indexs)
        self.tracker_last = self.tracker_new

    def load_yolov3_model(self):
        yolo_model_path = os.path.expanduser(self.yolo_model_path)
        assert yolo_model_path.endswith(".h5")

        self.yolo_model = load_model(yolo_model_path, compile=False)
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = k.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.yolo_score, iou_threshold=self.yolo_iou)
        return boxes, scores, classes

    def construct_RNN_Model(self, pretrained=True):
        inputs_g = Input(shape=(3, 6))
        x_g = GRU(128, return_sequences=True, unroll=True)(inputs_g)
        x_g = GRU(128, return_sequences=True, unroll=True)(x_g)
        x_g = GRU(64, return_sequences=True, unroll=True)(x_g)
        # x_g = Dropout(0.1)(x_g)
        # x_g = GRU(64,return_sequences=True,unroll=True)(x_g)
        logits_distance_g = Dense(1, activation="relu", name="distance")(x_g)

        self.DisNet_Model = Model(inputs=inputs_g, outputs=logits_distance_g)
        # load DisNet pretrained weight
        if pretrained:
            DisNet_weights_path = os.path.expanduser(self.DisNet_weights_path)
            assert DisNet_weights_path.endswith(".h5")
            if os.path.isfile(DisNet_weights_path):
                self.DisNet_Model.load_weights(DisNet_weights_path)
                print("load pretrained DisNet Model")

    def zoom_in_image(self, image):
        image = np.asarray(image)
        image_center = np.divide(image.shape[:2], 2)
        resize_height = np.round(image.shape[0] * float(1.0 / self.zoom_in_factor))
        resize_width = np.round(image.shape[1] * float(1.0 / self.zoom_in_factor))

        width_begin = np.int(image_center[1] - resize_width / 2)
        width_end = np.int(image_center[1] + resize_width / 2)
        height_begin = np.int(image_center[0] - resize_height / 2)
        height_end = np.int(image_center[0] + resize_height / 2)
        image_crop = image[height_begin:height_end, width_begin:width_end]
        image_zoom = cv2.resize(image_crop, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(image_zoom)

    def detect_image(self, image):
        image = np.asarray(image)
        image_center = np.divide(image.shape[:2], 2)
        resize_height = np.round(image.shape[0] * float(1.0 / self.zoom_in_factor))
        resize_width = np.round(image.shape[1] * float(1.0 / self.zoom_in_factor))

        width_begin = np.int(image_center[1] - resize_width / 2)
        width_end = np.int(image_center[1] + resize_width / 2)
        height_begin = np.int(image_center[0] - resize_height / 2)
        height_end = np.int(image_center[0] + resize_height / 2)
        image_crop = image[height_begin:height_end, width_begin:width_end]
        image_zoom = cv2.resize(image_crop, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(image_zoom)

    def detect_image(self, image):
        start = time.time()
        self.img_width = image.size[0]
        self.img_height = image.size[1]
        # Display Settings
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(1.2e-2 * image.size[1] + 0.5).astype('int32'))
        font_corner = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                         size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        detect_thickness = (image.size[0] + image.size[1]) // 900
        track_thickness = (image.size[0] + image.size[1]) // 700

        # Check camera is zoomed or not
        if not self.ZoomCamera:
            if self.zoom_in_factor != 1:
                image = self.zoom_in_image(image)
        print(image.size)

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
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                k.learning_phase(): 0
            })
        detect_classes = []
        detect_index = []
        detect_classes_name = []
        detect_boxes = []
        index_current = 0
        self.detection_box_yolo = out_boxes
        for i, c in reversed(list(enumerate(out_classes))):
            if self.class_names[c] in self.detect_classes:
                this_class = self.class_names[c]
                this_box = out_boxes[i]

                # add detection result in Array
                detect_classes.append(c)
                detect_index.append(index_current)
                detect_classes_name.append(this_class)
                detect_boxes.append(this_box)
                index_current += 1

        ### update tracker
        self.update_tracker(detect_boxes, detect_classes_name, detect_classes, np.array(detect_index))

        ### run tracker to predict result in next frame and distance
        if len(self.tracker_last):
            self.DisNet_input_reshape = self.tracker_last[:, :, :6].reshape(-1, 6)
            self.DisNet_input_scaled = self.scaler.transform(self.DisNet_input_reshape)
            self.DisNet_input = self.DisNet_input_scaled.reshape(-1, 3, 6)
            self.distances = self.DisNet_Model.predict(x=self.DisNet_input)

        draw = ImageDraw.Draw(image)
        label_title = '{0:^10}{1:^10}{2:^10}{3:^10}'.format('Class', 'Index', 'Distance', 'Position')
        title_size_corner = draw.textsize(label_title, font_corner)
        title_corner = np.array([image.size[0] - title_size_corner[0], 0])

        draw.rectangle(
            [tuple(title_corner), tuple(title_corner + title_size_corner)],
            fill=(200,200,200))
        draw.text(title_corner, label_title, fill=(0,0,0), font=font_corner)

        del draw

        ### draw tracking result
        for i in range(len(self.tracker_last)):
            draw = ImageDraw.Draw(image)
            # generate tracking distance and box
            this_detect_index = int(self.tracker_detect_index[i])
            distance = np.squeeze(self.distances[i, -1]) * self.zoom_in_factor

            box = detect_boxes[this_detect_index]

            ### get tracker class
            this_tracker_class = self.tracker_class[i]
            this_tracker_class_name = self.class_names[this_tracker_class]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            height = bottom - top
            height_class = self.class_size_dict[this_tracker_class_name][1]
            pixel_meter = float(height_class) / height
            position = float(self.img_width - (right + left)) / 2 * pixel_meter

            ### Setting label
            label_left = '{0:^10}{1:^10}{2:^10.2f}{3:^10.2f}'.format(this_tracker_class_name, this_detect_index,
                                                                     float(distance), position)
            label_size_corner = draw.textsize(label_left, font_corner)

            ### draw detection result
            label = '{}{}'.format(this_tracker_class_name, this_detect_index)
            label_size = draw.textsize(label, font)

            corner = np.array([image.size[0] - title_size_corner[0], title_size_corner[1] * (i + 1)])

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])

            else:
                text_origin = np.array([left, top + 1])

            for t in range(detect_thickness):
                draw.rectangle(
                    [left + t, top + t, right - t, bottom - t],
                    outline=self.yolo_class_colors[this_tracker_class])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.yolo_class_colors[this_tracker_class])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            draw.rectangle(
                [tuple(corner), tuple(corner + title_size_corner)],
                fill=self.yolo_class_colors[this_tracker_class])
            draw.text(corner, label_left, fill=(0, 0, 0), font=font_corner)
            del draw

        end = time.time()
        print("FPS: {:.2f}".format(1.0 / (end - start)))

        return image

    def close_session(self):
        self.sess.close()
