#!/usr/bin/env python
import sys
import os
import cv2
import numpy as np
from color_detector import get_block_detections

#Following Code uses Python3 so uncommenting leaves an error when running on Robot
# Reading the video from the
# webcam in image frames
file_name = "lot_blocks_light_Color.png"
imageFrame = cv2.imread(f"./{file_name}")
print(imageFrame.shape)
# Convert the imageFrame in BGR(RGB color space) to
# HSV(hue-saturation-value) color space
#hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
# Set range for red color and define mask
imageFrame, blocks = get_block_detections(imageFrame)
print(imageFrame.shape)
cv2.imwrite(f"final_annotated_{file_name}", imageFrame)