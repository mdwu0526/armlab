#!/usr/bin/python3
import os

# Python code for Multiple Color Detection
import numpy as np
import cv2
import itertools



def get_block_detections(imageFrame):

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([120, 180, 75], np.uint8)
    red_upper = np.array([200, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
    green_lower = np.array([40, 130, 50], np.uint8) #120 best
    green_upper = np.array([90, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    blue_lower = np.array([95, 200, 80], np.uint8)
    blue_upper = np.array([110, 255, 160], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    yellow_lower = np.array([20, 180, 100], np.uint8) #
    yellow_upper = np.array([40, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    orange_lower = np.array([5, 145, 120], np.uint8)
    orange_upper = np.array([20, 255, 250], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    purple_lower = np.array([110, 70, 30], np.uint8)
    purple_upper = np.array([170, 130, 140], np.uint8)
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)

    pink_lower = np.array([150, 50, 70], np.uint8)
    pink_upper = np.array([200, 155, 100], np.uint8)
    pink_mask = cv2.inRange(hsvFrame, pink_lower, pink_upper)

    kernal = np.ones((4, 4), "uint8")

    red_mask = cv2.dilate(red_mask, kernal)
    green_mask = cv2.dilate(green_mask, kernal)
    blue_mask = cv2.dilate(blue_mask, kernal)
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    orange_mask = cv2.dilate(orange_mask, kernal)
    purple_mask = cv2.dilate(purple_mask, kernal)
    pink_mask = cv2.dilate(pink_mask, kernal)

    masks = [red_mask, green_mask, blue_mask, yellow_mask, orange_mask, purple_mask]#, pink_mask]
    color_names = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple"]#, "Pink"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 128, 255), (255, 0, 128)]#, (255, 0, 255)]
    all_blocks = []
    for mask, color_name, color in zip(masks, color_names, colors):
        imageFrame, blocks = put_bounding_boxes(imageFrame, mask, color_name, color)
        all_blocks.append({color_name: blocks}) #FOR COLOR #This is list of dictionaries where value is a list
    return imageFrame, all_blocks



def put_bounding_boxes(imageFrame,mask, color_name, color):
    # Creating contour to track green color
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    blocks = []
    HIGH = 100
    LOW = 10
    RATIO = 1.6
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            # print(rect[0]) #This is the center (x,y)
            # print(rect[1]) #This is the (height,width)
            # print(rect[2]) #This is angle [-90, 0]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if (w < HIGH and w > LOW and h < HIGH and h > LOW): #Is Valid block size..MAY NEED TO CHANGE
                blocks.append(rect)
                imageFrame = cv2.drawContours(imageFrame, [box], 0, color, 2)
                blockSizeInfo = str(w) + " " + str(h) + " " + str(w*h)
                cv2.putText(imageFrame, blockSizeInfo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                x = int(rect[0][0])
                y = int(rect[0][1])
                cv2.circle(imageFrame, (x,y), 1, color, -1)
    
    for a, b in itertools.combinations(blocks, 2):
        diff = ((a[0][0] - b[0][0])**2 + (a[0][1] - b[0][1])**2)**0.5
        if diff < 28:
            if b in blocks:
                blocks.remove(b)  #Same center so eliminate it
    return imageFrame, blocks