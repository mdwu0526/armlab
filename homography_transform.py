#!/usr/bin/python
""" Example: 

python find_contours_in_depth.py -i image_blocks.png -d depth_blocks.png -l 905 -u 973

"""
import argparse
import sys
import cv2
import numpy as np

def homography_transform_helper(image):
    """
    bruh
    """

    # Select source points to apply the homography transform from
    src_pts = np.array([167, 93, 231, 663, 1089, 645, 1122, 66]).reshape((4,2))

    # Draw circles on the original image at the specified points
    for pt in src_pts:
        cv2.circle(image, tuple(pt), 5, (0, 255, 0), -1)

    # Select destination points to apply the homography transform to
    dest_pts = np.array([100, 100, #    Top Left
                    100, 650, #         Bot Left
                    1200, 650, #         Bot Right
                    1200, 100, #         Top Right
                    ]).reshape((4, 2))

    H = cv2.findHomography(src_pts, dest_pts)[0]

    new_img = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

    # Return image
    return new_img, H


def homography_transform_helper_april_tags(image, tags):
    """
    bruh
    """

    tags = tags.reshape(4,2)

    # Draw circles on the original image at the specified points
    for pt in tags:
        cv2.circle(image, tuple(pt), 5, (255, 0, 0), -1)

    # Select destination points to apply the homography transform to
    dest_pts = np.array([
                    375, 550, #         Bot Left
                    925, 550, #         Bot Right
                    925, 230, #         Top Right
                    375, 230, #         Top Left
                    ]).reshape((4, 2))

    H = cv2.findHomography(tags, dest_pts)[0]

    new_img = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

    # Return image
    return new_img, H