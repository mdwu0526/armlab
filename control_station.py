#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import sys
import cv2
import numpy as np
import rospy
import time
from functools import partial

from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
from PyQt4.QtGui import (QPixmap, QImage, QApplication, QWidget, QLabel,
                         QMainWindow, QCursor, QFileDialog)

from copy import deepcopy
from ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread

""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi
DEPTH = 995 # Measured from camera sensor to base board (mm)

class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        if (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        # Provided User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.open_gripper())
        self.ui.btnUser2.clicked.connect(lambda: self.set_gripper(False))
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(lambda: self.rxarm.close_gripper())
        self.ui.btnUser3.clicked.connect(lambda: self.set_gripper(True))
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))

        # Teach and Repeat
        self.ui.btnUser5.setText("Teach")
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'teach'))
        nxt_if_arm_teach = lambda next_state: self.sm.set_next_state(next_state)

        # Save Waypoint
        self.ui.btnUser6.setText("Save Waypoint")
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_teach, 'save_waypoint'))

        # Clear Previous Waypoint
        self.ui.btnUser7.setText("Clear Previous Waypoint")
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_teach, 'clear_waypoint'))

        # Print Saved Waypoints
        self.ui.btnUser8.setText("Print Saved Waypoints")
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_teach, 'print_waypoints'))       

        # Stop Motion
        self.ui.btnUser9.setText("Stop Motion")
        self.ui.btnUser9.clicked.connect(partial(nxt_if_arm_teach, 'stop_motion'))

        #Event 1
        self.ui.btnUser10.setText("Pick n sort")
        self.ui.btnUser10.clicked.connect(partial(nxt_if_arm_init, 'pick_n_sort'))
        
        #Event 2
        self.ui.btnUser11.setText("Pick n stack")
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'pick_n_stack'))

        #Event 3
        self.ui.btnUser12.setText("Line em up")
        self.ui.btnUser12.clicked.connect(partial(nxt_if_arm_init, 'line_up'))

        # #Event 4
        self.ui.btnUser13.setText("Stack em high")
        self.ui.btnUser13.clicked.connect(partial(nxt_if_arm_init, 'stack_high'))



        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Pick and Place
        self.ui.chk_pickandplace.stateChanged.connect(self.pickAndPlaceChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

        # Cause block detection logic to run every N frames
        self.counter = 0



    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f rad" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f rad" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image, block_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """

        ### Additional every-frame code ###

        # Get intrinsic and extrinsic matricies
        inM = self.sm.camera_intrinsic
        exM = self.sm.camera_extrinsic
        self.camera.intrinsic_matrix = inM
        self.camera.extrinsic_matrix = exM

        # Pass stuff to camera
        if self.camera.calibrated != self.sm.calibrated:
            self.camera.detected_tag_locations = self.sm.board_tag_image_points
            self.camera.calibrated = self.sm.calibrated

        # Detect blocks
        self.sm.detect()

        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))
        if (self.ui.radioUsr3.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(block_image))

    """ Other callback functions attached to GUI elements"""
    
    def qimg2cv(self, q_img):
        q_img.save('temp.png', 'png')
        mat = cv2.imread('temp.png')
        return mat

    def cv2qimg(self, cvImg):
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def set_gripper(self, closed):
        self.sm.gripper_closed = closed

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)
            self.ui.chk_pickandplace.setChecked(False)

    def pickAndPlaceChk(self, state):
        """!
        @brief      Changes to pick and place mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            self.sm.set_next_state("mousemode")
            self.ui.videoDisplay.mousePressEvent = self.worldCoordMousePress
            self.sm.gripper_closed = False

        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.chk_directcontrol.setChecked(False)
            self.ui.chk_pickandplace.setChecked(False)


    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    After implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (pt.x(), pt.y(), z))

            # Get intrinsic and extrinsic matricies
            inM = self.sm.camera_intrinsic
            #exM = np.array([[1, 0, 0, xAdditionalOffset], [0, np.cos(angle_radians), -np.sin(angle_radians), yOffset+yAdditionalOffset], [0, np.sin(angle_radians), np.cos(angle_radians),	zOffset], [0,	0, 0, 1]])
            exM = self.sm.camera_extrinsic
            self.camera.intrinsic_matrix = inM
            self.camera.extrinsic_matrix = exM

            # Get mouse points [u,v,1]
            pt_np = np.array([pt.x(), pt.y(), 1])

            if (self.camera.homography_matrix is not None) and (self.camera.calibrated == True):
                testMat = pt_np
                testMat = np.matmul(np.linalg.inv(self.camera.homography_matrix),testMat)
                
                pt_np[0] = testMat[0]/testMat[2]
                pt_np[1] = testMat[1]/testMat[2]
                pt_np[2] = testMat[2]/testMat[2]

                if pt_np[0] > 1280:
                    pt_np[0] = 1280
                if pt_np[0] < 0:
                    pt_np[0] = 0
                if pt_np[1] < 0:
                    pt_np[1] = 0
                if pt_np[1] > 720:
                    pt_np[1] = 720

                z = (cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.depth_transform_matrix, (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0])))[pt_np[1]][pt_np[0]]

            # Multiply inverse of intrinsic matrix by mouse points and scale by z
            coord_c = np.matmul(np.linalg.inv(inM),pt_np)
            coord_c = coord_c * z

            # Convert to Homogeneous coordinates
            coord_c = np.append(coord_c, 1)

            # Multiply inverse of extrinsic matrix by Homogeneous coordinates to get world coordinates
            [self.sm.u,self.sm.v,self.sm.w,_] = np.matmul(np.linalg.inv(exM),coord_c)

            # Scale Y range to get Z adjustment factor
            NewValue = 0.5
            OldRange = (475 - (-175))
            if (OldRange is not 0):
                NewRange = (1 - 0)  
                NewValue = (((self.sm.v - -175) * NewRange) / OldRange) + 0
            if (NewValue < 0):
                NewValue = 0
            elif NewValue > 1:
                NewValue = 1
            NewValue = 1-NewValue
            self.sm.w = self.sm.w + 20*NewValue

            # Print world coordinates
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                             (self.sm.u,self.sm.v,self.sm.w))
            
            # Pass stuff to camera
            if self.camera.calibrated != self.sm.calibrated:
                self.camera.detected_tag_locations = self.sm.board_tag_image_points
                self.camera.calibrated = self.sm.calibrated

    def worldCoordMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions and saves it to global coordinates

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        if self.sm.current_state == "mousemode":
            self.sm.clicked_u = self.sm.u
            self.sm.clicked_v = self.sm.v
            self.sm.clicked_w = self.sm.w
            print('Click test',self.sm.clicked_u,self.sm.clicked_v,self.sm.clicked_w)
            self.sm.next_state = "pcknplce"
        else:
            print("Can not choose waypoint yet!")


    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.ui.chk_pickandplace.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dh_config_file=args['dhconfig'])
    app_window.show()
    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=True,
                    help="path to DH parameters csv file")
    main(args=vars(ap.parse_args()))
