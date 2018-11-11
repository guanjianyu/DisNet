import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from PIL import Image as Image_PIL

from DisNetRNN_2 import DisNet_RNN
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--ZoomCamera', type=int, help='type of Camera, Zooming camera is True', default=0)
parser.add_argument('--ZoomFactor', type=int, help='Zooming factor of choosing camera', default=1)
args = parser.parse_args()


class ObjectDetect:

    def __init__(self):
        self.sub = rospy.Subscriber("/cam02/camera/image_raw", Image, self.callback)  # thermal_camera/image
        self.cv_bridge = CvBridge()

    def callback(self, data):
        global i
        try:
            image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        image = Image_PIL.fromarray(image)
        with graph.as_default():
            image_zoom = model.detect_image(image)
        result = np.asarray(image_zoom)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        cv2.waitKey(1)


if __name__ == '__main__':
    import tensorflow as tf

    rospy.init_node("object_crop_detection", anonymous=True)
    print(args.ZoomCamera)
    print(args.ZoomFactor)

    model = DisNet_RNN(bZoomCamera=args.ZoomCamera, scale=args.ZoomFactor)
    model.construct_RNN_Model()
    graph = tf.get_default_graph()
    ob = ObjectDetect()
    i = 1

    try:

        rospy.spin()
        model.close_session()
    except KeyboardInterrupt:
        print("shut down")
    cv2.destroyAllWindows()
