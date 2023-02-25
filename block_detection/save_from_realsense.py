#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
import argparse
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageListener:
    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic,Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        (rows,cols,channels) = cv_image.shape
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image window", cv_image)
        global save_image
        global save_depth
        k = cv2.waitKey(1)
        if k == ord('s'): # wait for 's' key to save and exit    
            save_depth = True
            save_image = True
        if save_image == True:
            cv2.imwrite(image_file, cv_image)
            print(image_file + " saved.")
            save_image = False

class DepthListener:
    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic,Image,self.callback)

    def callback(self,data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        (rows,cols) = cv_depth.shape
        clipped = np.clip(cv_depth, 0, 2000).astype(np.uint8)
        normed = cv2.normalize(clipped, clipped, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_depth = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        cv2.imshow("Depth window", color_depth)
        global save_image
        global save_depth
        k = cv2.waitKey(1)
        if k == ord('s'): # wait for 's' key to save and exit    
            save_depth = True
            save_image = True
        if save_depth == True:
            cv2.imwrite(depth_file, cv_depth)
            print(depth_file + " saved.")
            save_depth = False

save_image = False
save_depth = False
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "Base filename for the images")
args = vars(ap.parse_args())
image_file = str("image_" + args["output"] + ".png")
depth_file = str("depth_" + args["output"] + ".png")
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
image_topic = "/camera/color/image_raw"
depth_topic = "/camera/aligned_depth_to_color/image_raw"
image_listener = ImageListener(image_topic)
depth_listener = DepthListener(depth_topic)
rospy.init_node('realsense_viewer', anonymous=True)
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()