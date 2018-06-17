import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from PIL import Image as Image_PIL

from DisNet import DisNet

class Object_crop:

    def __init__(self):
        self.sub = rospy.Subscriber("/cam01/camera/image_raw", Image, self.callback)
        self.cv_bridge = CvBridge()

    def callback(self,data):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        image = Image_PIL.fromarray(image)
        with graph.as_default():
            image_zoom = model.detect_image(image)
            result = np.asarray(image_zoom)
            cv2.namedWindow("result",cv2.WINDOW_NORMAL)
            cv2.imshow("result",result)
            cv2.waitKey(1)

if __name__ == '__main__':
    import tensorflow as tf

    rospy.init_node("object_crop_detection", anonymous=True)
    model=DisNet(bZoomCamera=False,scale=5)
    graph = tf.get_default_graph()
    ob = Object_crop()
    try:
        rospy.spin()
        model.close_session()
    except KeyboardInterrupt:
        print("shut down")
    cv2.destroyAllWindows()
