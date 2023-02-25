"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
from homography_transform import homography_transform_helper, homography_transform_helper_april_tags
from color_detector import get_block_detections


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.BlockFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])
        self.homography = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.counter_time = 0

        self.calibrated = False
        self.detected_tag_locations = None
        self.homography_matrix = None

        # Original points from raised April tags
        #self.depth_points = np.float32([[1067,620],[606,210],[546,466],[179,138]])
        #self.depth_image_points = np.float32([[1078,625],[604,204],[516,463],[173,133]])

        # New points from board corners
        self.depth_points = np.float32([[167,103],[1117,72],[1090,638],[236,658]])
        self.depth_image_points = np.float32([[161,94],[1128,61],[1100,641],[224,665]])
        
        self.depth_transform_matrix = cv2.getPerspectiveTransform(self.depth_points, self.depth_image_points)

        # Factory calibration
        self.intrinsic_matrix = np.array([[900.7150269,0,652.2869263],[0,900.1925049,358.3596191],[0,0,1]])

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = None
            if self.calibrated:
                frame, self.homography_matrix = homography_transform_helper_april_tags(self.VideoFrame, self.detected_tag_locations)
            else:
                frame = self.VideoFrame
            frame = cv2.resize(frame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                        QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtBlockFrame(self, frame):
        try:
            if self.calibrated:
                frame, self.homography_matrix = homography_transform_helper_april_tags(self.VideoFrame, self.detected_tag_locations)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame, self.block_detections = get_block_detections(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = self.VideoFrame
            frame = cv2.resize(frame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                        QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:

            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            self.DepthFrameRGB = cv2.warpPerspective(self.DepthFrameRGB, self.depth_transform_matrix, (self.DepthFrameRGB.shape[1], self.DepthFrameRGB.shape[0]))
            if self.calibrated:
                self.DepthFrameRGB = cv2.warpPerspective(self.DepthFrameRGB, self.homography_matrix, (self.DepthFrameRGB.shape[1], self.DepthFrameRGB.shape[0]))
                #frame = self.VideoFrame
            else:
                pass
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        _, self.block_detections = get_block_detections(self.camera.convertQtVideoFrame())

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """

        # Copy VideoFrame to GridFrame
        self.GridFrame = np.copy(self.VideoFrame)

        # If calibrated, homography it
        doHomography = False
        try:
            #TODO: Apply Homography to the frame
            frame = None
            if self.calibrated:
                frame, self.homography_matrix = homography_transform_helper_april_tags(self.GridFrame, self.detected_tag_locations)
                doHomography = True
            else:
                #frame, self.homography_matrix = homography_transform_helper(self.VideoFrame)
                frame = self.VideoFrame
        except:
            pass

        # Matrix for image points aligning with grid points
        camera_points = []
        image_points = []

        # For every grid point in the world frame
        for gridPointX in self.grid_x_points:
            for gridPointY in self.grid_y_points:
                # Multiplies the X, Y, 1 grid point positions with the camera intrinsic matrix and divide by the depth
                # to return (u,v,1). See transformation lecture slides for more details
                world_points = [gridPointX, gridPointY, 1, 1]
                camera_points = np.append(camera_points, np.matmul(self.extrinsic_matrix,world_points))
        
        # Resize camera points to ensure it's a list of points
        camera_points = np.reshape(camera_points,((self.grid_x_points.shape[0]*self.grid_y_points.shape[0]),4))

        for cameraPoint in camera_points:
            imagePoint = np.matmul(self.intrinsic_matrix,[cameraPoint[0],cameraPoint[1],cameraPoint[2]])
            imagePoint[0] = imagePoint[0]/imagePoint[2]
            imagePoint[1] = imagePoint[1]/imagePoint[2]
            imagePoint[2] = imagePoint[2]/imagePoint[2]
            image_points = np.append(image_points, imagePoint)

        # Resize image points to ensure it's a list of points
        image_points = np.reshape(image_points,((self.grid_x_points.shape[0]*self.grid_y_points.shape[0]),3))

        #image_coords = np.matmul(self.homography_matrix,[gridPointX,gridPointY,1])
        for imagePoint in image_points:
            cv2.circle(self.GridFrame, tuple((int(imagePoint[0]),int(imagePoint[1]))), 5, (0, 0, 255), -1)

        if doHomography:
            self.GridFrame = cv2.warpPerspective(self.GridFrame, self.homography_matrix, (self.GridFrame.shape[1], self.GridFrame.shape[0]))
        
        

class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Block Detect window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_frame = self.camera.convertQtBlockFrame(rgb_frame)
            if block_frame is None:
                block_frame = rgb_frame
            self.camera.projectGridInRGBImage()
            grid_frame = self.camera.convertQtGridFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(
                    rgb_frame, depth_frame, tag_frame, grid_frame, block_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Grid window",
                    cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Block Detect window",
                    cv2.cvtColor(self.camera.BlockFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
