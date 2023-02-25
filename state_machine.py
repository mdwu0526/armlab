"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
from std_msgs.msg import String
import cv2
from camera import TagDetectionListener
from solve_extrinsics import recover_homogenous_transform_pnp
from kinematics import IK_geometric, jointCheck
from copy import deepcopy

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """



        # APRIL TAG SETUP - MUST USE IN ASCENDING ORDER!!!
        self.numPoints = 4  # Number of April tags
        self.april_tags = np.array([
                        -250, -25, 0, #     ID 1
                        250, -25, 0, #      ID 2
                        250, 275, 0, #      ID 3
                        -250, 275, 0, #     ID 4
                        -425, 400, 90, #    ID 5
                        325, -100, 150, #   ID 6
                        150, 325, 242 #     ID 7
                        ])
        
        # Factory Camera Intrinsics + Distortion:
        self.camera_intrinsic = np.array([[900.7150269,0,652.2869263],[0,900.1925049,358.3596191],[0,0,1]])
        self.distortion_coef = np.array([0.1490122675895691, -0.5096240639686584, -0.0006352968048304319, 0.0005230441456660628, 0.47986456751823425])
        
        # Camera Extrinsics
        self.camera_extrinsic = np.array([[-0.99919277,0.02840217,-0.02840989,3.70212455],
                                            [0.03340087,0.98029375,-0.19470119,-150.01502800],
                                            [0.02232010,-0.19549294,-0.98045108,-936.29050200],
                                            [0.00000000,0.00000000,0.00000000,1.00000000]])

        self.large_block = True    
        self.PSI = 0   

        self.rxarm = rxarm
        self.src_pts_pos = None
        self.src_pts_ori = None
        self.gripper_closed = False
        self.saved_waypoints = None
        self.bigR_waypoints = [np.array([0.96333998,0.39269909,0.49394181,-1.59227204,-0.56910688]),
                                np.array([0.96027201,0.4954758,0.27151459,-1.32996142,-0.57217485]),
                                np.array([0.96333998,0.39269909,0.49394181,-1.59227204,-0.56910688])]
        self.bigO_waypoints = [np.array([0.8084079,0.14572819,0.11658254,-1.50943708,-0.74858266]),
                                np.array([0.81147587,0.2316311,-0.14726216,-1.18883514,-0.76238847]),
                                np.array([0.8084079,0.14572819,0.11658254,-1.50943708,-0.74858266])]
        self.bigY_waypoints = [np.array([0.58904862,-0.0966408,-0.19634955,-1.44194198,-0.97714579]),
                                np.array([0.58444667,0.00306796,-0.46786416,-1.06918466,-1.01396132]),
                                np.array([0.58904862,-0.0966408,-0.19634955,-1.44194198,-0.97714579])]
        self.bigG_waypoints = [np.array([0.27458256,-0.17947575,-0.2960583,-1.4787575,-1.41433036]),
                                np.array([0.27765054,-0.11504856,-0.61205834,-1.05537879,-1.3713789,]),
                                np.array([0.27458256,-0.17947575,-0.2960583,-1.4787575,-1.41433036])]
        self.bigB_waypoints = [np.array([-0.11965051,-0.27765054,-0.38656318,-1.43427205,-1.68737888]),
                                np.array([-0.11351458,-0.16106799,-0.6657477,-1.03543711,-1.68431091]),
                                np.array([-0.11965051,-0.27765054,-0.38656318,-1.43427205,-1.68737888])]
        self.bigV_waypoints = [np.array([-0.44178647,-0.20248547,-0.31139812,-1.4296701,-2.01565075]),
                                np.array([-0.45405832,-0.05675729,-0.54916513,-1.1044662,-1.99417508]),
                                np.array([-0.44178647,-0.20248547,-0.31139812,-1.4296701,-2.01565075])]
        self.place_gripper_pos = [True, False, False]
        self.bigSlide_waypoints =  [np.array([-0.88510692,0.2316311,0.22089323,-1.61221385,-0.88970888]),
                                    np.array([-0.7593205,0.09970875,-0.35588354,-1.16275752,-0.87897104]),
                                    np.array([-0.22549519,-0.07669904,-0.57831079,-1.18883514,-0.21935926]),
                                    np.array([-0.55836904,-0.15953401,-0.28532043,-1.38671863,-0.49700978]),
                                    np.array([1.06611669,0.73784477,1.07838857,-1.8637867,1.04157293]),
                                    np.array([1.0937283,0.75625253,0.73170888,-1.36831093,1.05077684]),
                                    np.array([0.9265244,0.37582532,0.06749516,-1.282408,0.9372623]),
                                    np.array([1.00015545,0.46479619,0.59211659,-1.66283524,0.98788363])]
        self.smallR_waypoints = [np.array([0.38963112,0.38042724,0.46939814,-1.6275537,-1.13514578]),
                                np.array([0.38349521,0.47553405,0.12732041,-1.19497108,-1.14741766]),
                                np.array([0.38963112,0.38042724,0.46939814,-1.6275537,-1.13514578])]
        self.smallO_waypoints = [np.array([0.23316509,0.27304858,0.30219424,-1.55852449,-1.28547597]),
                                np.array([0.24236897,0.40803891,0.00460194,-1.1642915,-1.33609736]),
                                np.array([0.23316509,0.27304858,0.30219424,-1.55852449,-1.28547597])]
        self.smallY_waypoints = [np.array([0.09357283,0.2239612,0.22702916,-1.53551483,-1.48335946]),
                                np.array([0.08743691,0.38963112,-0.02761165,-1.16582549,-1.47262156]),
                                np.array([0.09357283,0.2239612,0.22702916,-1.53551483,-1.48335946])]
        self.smallG_waypoints = [np.array([-0.08436894,0.23469907,0.24236897,-1.53398085,-1.62908769]),
                                np.array([-0.08590293,0.37582532,-0.05522331,-1.13974774,-1.67050517]),
                                np.array([-0.08436894,0.23469907,0.24236897,-1.53398085,-1.62908769])]
        self.smallB_waypoints = [np.array([-0.24850489,0.24236897,0.25310683,-1.518641,-1.81316531]),
                                np.array([-0.24850489,0.42031074,0.02607767,-1.1934371,-1.80242753]),
                                np.array([-0.24850489,0.24236897,0.25310683,-1.518641,-1.81316531])]
        self.smallV_waypoints = [np.array([-0.40343696,0.33900976,0.39730105,-1.58153427,-1.92514598]),
                                np.array([-0.398835,0.46019426,0.08897088,-1.1826992,-1.923612,]),
                                np.array([-0.40343696,0.33900976,0.39730105,-1.58153427,-1.92514598])]
        self.smallSlide_waypoints =  [np.array([-0.65194184,0.73631078,1.0937283,-1.85918474,-0.60132051]),
                                    np.array([-0.63200009,0.72403896,0.64273798,-1.44654393,-0.59978652]),
                                    np.array([-0.31906801,0.38196123,0.03528156,-1.18423319,-0.30372819]),
                                    np.array([-0.53535932,0.42491269,0.54302919,-1.59073818,-0.47706804]),
                                    np.array([0.61972827,0.60745639,0.85749531,-1.64289343,0.56910688]),
                                    np.array([0.65807778,0.77926224,0.74551469,-1.52017498,0.72557294]),
                                    np.array([0.34974763,0.39423308,0.04141748,-1.16889334,0.42337871]),
                                    np.array([0.63660204,0.62126225,0.88510692,-1.68891287,0.57831079])]
        self.reset_waypoints =  [np.array([-0.01227185,0.01073787,-0.0644272,-0.00920388,0.0]),
                                np.array([-0.01533981,-1.80089355,-1.5677284,-0.80380595,0.0])]
        self.slide_gripper_pos = [False,False,False,False,False,False,False,False]
        self.reset_gripper_pos = [False,False]
        # self.saved_waypoints = [np.array([-0.0076699 ,  0.00920388, -0.05982525, -0.01073787,  0.        ]), np.array([-0.0076699 ,  0.00920388, -0.05982525, -0.01073787,  0.        ]), np.array([-0.0076699 ,  0.0076699 , -0.05982525, -0.01073787,  0.        ]), np.array([-0.52615541,  0.08743691, -0.15646605, -0.99862152,  0.        ]), np.array([-0.5276894 ,  0.10891264, -0.15646605, -0.99708754, -0.3666214 ]), np.array([-0.52615541,  0.34054375, -0.07363108, -0.99708754, -0.3666214 ]), np.array([-0.52462143,  0.34054375, -0.06596117, -0.99862152, -0.3666214 ]), np.array([-0.5276894 ,  0.18254372, -0.06596117, -0.99862152, -0.3666214 ]), np.array([-1.18730116,  0.18100974, -0.06596117, -0.99862152, -0.3666214 ]), np.array([-1.18730116,  0.27304858, -0.1672039 , -0.99708754, -0.3666214 ]), np.array([-1.18730116,  0.27151459, -0.1672039 , -0.99708754, -0.3666214 ]), np.array([-1.18730116, -0.22089323, -0.1672039 , -0.99862152, -0.3666214 ]), np.array([ 0.35741752, -0.22089323, -0.1672039 , -0.99862152, -0.3666214 ]), np.array([ 0.34207773,  0.02147573, -0.1672039 , -0.99862152,  0.36508745]), np.array([ 0.34207773,  0.32980588, -0.079767  , -0.99708754,  0.36508745]), np.array([ 0.34207773,  0.32520393, -0.079767  , -0.99862152,  0.36508745]), np.array([ 0.34207773,  0.21629129, -0.08283497, -0.99862152,  0.36508745]), np.array([-0.495,  0.21629129, -0.08283497, -0.99862152, -0.3666214 ]), np.array([-0.495,  0.33287385, -0.08130098, -0.99708754, -0.3666214 ]), np.array([-0.495,  0.34054375, -0.08130098, -0.99708754, -0.36508745]), np.array([-0.48933989,  0.04295146, -0.08283497, -0.99708754, -0.3666214 ]), np.array([-1.18116522,  0.04295146, -0.08283497, -0.99862152, -0.3666214 ]), np.array([-1.16122353,  0.21935926, -0.08283497, -0.99708754, -0.3666214 ]), np.array([-1.20724297,  0.21322334, -0.23469907, -0.99708754, -0.3666214 ]), np.array([-1.18730116,  0.26844665, -0.18561168, -0.99708754, -0.3666214 ]), np.array([-1.18423319,  0.26844665, -0.18100974, -0.99862152, -0.3666214 ]), np.array([-1.18730116, -0.13805827, -0.18100974, -0.99862152, -0.3666214 ]), np.array([ 0.32366997, -0.13805827, -0.18100974, -0.99862152,  0.36508745]), np.array([ 0.33133987,  0.30526218, -0.10124274, -0.99708754,  0.36508745]), np.array([ 0.33133987,  0.30526218, -0.10124274, -0.99708754,  0.47093213]), np.array([ 0.33133987,  0.30526218, -0.10124274, -0.99708754,  0.47093213]), np.array([ 0.33133987, -0.12425245, -0.10124274, -0.99862152,  0.47093213]), np.array([ 0.0076699 ,  0.0076699 , -0.09970875, -0.01073787,  0.        ])]
        # self.saved_gripper_pos = [False, True, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, True, False, False, False]
        ##self.saved_gripper_pos = None
        self.camera = camera
        self.status_message = "State: Idle"
        self.prev_state = "idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        
        # Stores the teach command waypoints in an empty array
        self.teachwp = []

        # Is it calibrated
        self.calibrated = False
        self.board_tag_image_points = None
        self.tag_ids = None

        # Mouse coordinates in global coordinates saved
        self.u = None
        self.v = None
        self.w = None

        self.clicked_u = None
        self.clicked_v = None
        self.clicked_w = None

        self.desired_pose = None

        # Detected block coordinates in world coordinates
        self.detected_blocks = None

        # Base coordinates for keepout zone
        self.X_BASE_MIN = -100 
        self.X_BASE_MAX = 100
        self.Y_BASE_MIN = -175
        self.Y_BASE_MAX = 75

        self.z_offset_small = 0
        self.z_offset_large = 0

        self.stack_again = False
    
    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "pcknplce":
            self.pickAndPlace()

        if self.next_state == "mousemode":
            self.mouseMode()
        
        if self.next_state == "teach":
            self.teach()
        
        if self.next_state == "save_waypoint":
            self.save_waypoint()

        if self.next_state == "print_waypoints":
            self.print_waypoints()

        if self.next_state == "stop_motion":
            self.stop_motion()
        
        if self.next_state == "pick_n_sort":
            self.pick_n_sort()
        
        if self.next_state == "pick_n_stack":
            self.pick_n_stack()
        
        if self.next_state == "line_up":
            self.line_up()
        
        if self.next_state == "stack_high":
            self.stack_high_new()
        
        if self.next_state == "stack_rainbow":
            self.stack_rainbow()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def mouseMode(self):
        """!
        @brief Configures mouseMode
        """
        if (self.prev_state != "pcknplce"):
            self.status_message = "State: Mouse mode - Click on object of interest"
        self.current_state = "mousemode"
        self.PSI = 0

    # Moves the arm to a neutral position
    def set_neutral_joint_angle(self,desired_angle):
        """!
        @brief Saves a waypoint when the arm moves into its neutral position. Function must be called after an initial waypoint is set (can not append an empty array)
        """
        # Sets the position to the arm completely up
        self.saved_waypoints = np.vstack((self.saved_waypoints,np.array([desired_angle[0],-0.2623,1.1582,-0.4786,desired_angle[4]])))
        
        # Saves the current state of the gripper to the waypoint
        self.saved_gripper_pos = np.append(self.saved_gripper_pos, self.gripper_closed)

    def set_desired_joint_angle(self,waypoint,PSI,gripper_change):
        """!
        @brief Saves a waypoint given the XYZ position and PSI angle
        """
        # Saves all possible joint_angles as well as the wristpose
        joint_angles,_ = IK_geometric(waypoint,PSI)

        # Checks the limitations of the arm and returns the desired joint angles corresponding to whatever configuration is within actuator limitations
        desired_angle,_ = jointCheck(joint_angles)

        # If the PSI angle is zero, the wrist will not rotate (assuming blocks are in aligned on the grid)
        # TODO: Change how this is done based on the orientation of the blocks in block detector
        if PSI == np.pi/4:
            wrR = 0
        elif PSI == 0:
            wrR = 0
        else:
            wrR = -desired_angle[0]

        # Creates all five joint angles
        desired_angle = np.append(desired_angle, wrR)

        # Saves the joint angles into the global saved_waypoints list.
        if len(self.saved_waypoints) == 0:
            self.saved_waypoints = desired_angle
        else:
            self.saved_waypoints = np.vstack((self.saved_waypoints, desired_angle))
        
        # If the gripper needs to change state, change it and then store. Otherwise just store
        if gripper_change:
            self.gripper_closed = not self.gripper_closed
            self.saved_gripper_pos = np.append(self.saved_gripper_pos, self.gripper_closed)
        else:
            self.saved_gripper_pos = np.append(self.saved_gripper_pos, self.gripper_closed)

        return desired_angle


    def set_PSI(self,wristpose):
        """!
        @brief Sets the PSI angle of the arm and it will be constant throughout all the four waypoints
        """
        # If the wrist pose can drop over the POI, then it sets the PSI angle to 90 degrees. If not, the robot arm is set to 0 degrees
        if(np.sqrt(wristpose[0]**2 + wristpose[1]**2 + (wristpose[2]-103.91)**2) > 200+np.sqrt(50**2+200**2) or wristpose[2] > 180+174.15):
            PSI = self.PSI
        else:
            if self.PSI == np.pi/4:
                PSI = self.PSI
            else:
                PSI = np.pi/2

        return PSI

    def floor_check(self,gripper_horizontal,large_block):
        """!
        @brief If the arm is picking up the block, it will set the Z offset to the center of the block taking account into +/- 10mm of variation
        """
        # TODO: Check how much offset to configure if it detects a small or large block
        # Checks if the gripper state is in pick or place
        SCALE = 1.6
        LARGE_HEIGHT = 40
        SMALL_HEIGHT = 25
        # Large block offset
        if large_block:
            if self.gripper_closed and gripper_horizontal: # Place
                if self.PSI == np.pi/4:
                    OFFSET = 50
                else: 
                    OFFSET = 50
            elif self.gripper_closed and not gripper_horizontal: # Place
                OFFSET = 32
            elif not self.gripper_closed and gripper_horizontal: # Pick
                if self.PSI == np.pi/4:
                    OFFSET = 10
                else: 
                    OFFSET = 15
            else: # Pick
                OFFSET = 0
        # Small block offset
        else:
            if self.gripper_closed and gripper_horizontal: # Place
                OFFSET = 42
            elif self.gripper_closed and not gripper_horizontal: # Place
                OFFSET = 20
            elif not self.gripper_closed and gripper_horizontal: # Pick
                if self.PSI == np.pi/4:
                    OFFSET = 23
                else:
                    OFFSET = 25
            else: # Pick
                OFFSET = 12
    
        return OFFSET

    def pickAndPlace(self, mouse_mode = True):
        """!
        @brief      Configures to pick and place mode
        @operation  Takes the coordinate from mousemode and assigns four IK positions that the robot moves through to pick/place the block without knocking anything down
        """

        # Configures next state
        self.status_message = "Calculating IK and processing best joint angle..."
        self.current_state = "pcknplce"
        self.prev_state = self.current_state

        # Clear previous waypoints
        self.saved_waypoints = np.array([])
        self.saved_gripper_pos = np.array([])

        # Robot parameters
        PSI = None # Wrist angle with respect to the horizontal plane
        total_reach = 174.15+200+np.sqrt(50**2+200**2) # Total reach of the robot arm
        if mouse_mode:
            desired_pos = np.array([self.clicked_u, self.clicked_v, self.clicked_w]) # Stores clicked coordinates into a local variable
        else:
            desired_pos = self.desired_pose
        # Checks if the desired position can be reached by the span of the arm, if not, goes back to mousemode state for a new point to select
        if(np.sqrt(desired_pos[0]**2 + desired_pos[1]**2 + desired_pos[2]**2) > total_reach):
            self.status_message = "Can not reach this area!"
            self.next_state = "mousemode"
        else:
            
        ## WAYPOINT CODE BEGINS ##
            PSI = np.pi/2 # Define some arbitrary angle at first to find wristpose

            ## CHECKS IF THE WRIST CAN REACH OVER THE POINT OF INTEREST (POI) ##
            Z_OFFSET_WYPT1 = 80
            desired_angle,wristpose = IK_geometric(np.array([desired_pos[0],desired_pos[1],desired_pos[2]+Z_OFFSET_WYPT1]),PSI)
            PSI = self.set_PSI(wristpose) # Sets PSI for arm
            print("PSI:",PSI)
            print("self.PSI:",self.PSI)

            ## SETS THE FIRST POSITION OF THE ARM TO BE IN THE NEUTRAL POSITION ##
            # if len(self.saved_waypoints) == 0:
            #     self.set_neutral_waypoint(desired_angle)
            
            ## SETS THE SECOND POSITION OF THE ARM (RIGHT ABOVE POI) ##
            if PSI == np.pi/4:
                gripper_horizontal = True
            elif PSI == 0:
                # Sets the intial offset of the POI _cm in Z
                gripper_horizontal = True
            else: # If angle is 90, offset is a little different
                Z_OFFSET_WYPT1 = 65
                gripper_horizontal = False
            waypt1 = np.array([desired_pos[0],desired_pos[1],desired_pos[2]+Z_OFFSET_WYPT1])
            self.set_desired_joint_angle(waypt1,PSI,gripper_change = False)
            
            ## ASSIGNS ANOTHER Z OFFSET FOR THE THIRD POSITION OF THE ARM (IN POSITION TO PICK UP/PLACE THE BLOCK) ##
            Z_OFFSET_WYPT2 = self.floor_check(gripper_horizontal,self.large_block)
            # Place
            if self.gripper_closed:
                waypt2 = np.array([desired_pos[0],desired_pos[1],desired_pos[2]+Z_OFFSET_WYPT2])
            # Pick
            else:
                waypt2 = np.array([desired_pos[0],desired_pos[1],desired_pos[2]+Z_OFFSET_WYPT2])
            self.set_desired_joint_angle(waypt2,PSI,gripper_change = True)

            ## MOVES THE ARM SLIGHTLY AWAY FROM THE BLOCK DEPENDING ON PSI (POSITION FOUR) TO PREVENT KNOCKING OVER STACKS ##
            Z_OFFSET_WYPT3 = 30 # Tuning
            ANOTHER_OFFSET = 10
            dist = abs(Z_OFFSET_WYPT3*np.arctan2(desired_pos[0],desired_pos[1]))
            # TODO: FIX THIS LATER
            if desired_pos[1] > 0 and desired_pos[0] > 0:
                distx = -dist
                disty = -dist
            elif desired_pos[1] > 0 and desired_pos[0] < 0:
                distx = dist
                disty = -dist
            elif desired_pos[1] < 0 and desired_pos[0] < 0:
                distx = dist
                disty = dist
            else:
                distx = -dist
                disty = dist
            if PSI == np.pi/2:
                waypt3 = np.array([desired_pos[0],desired_pos[1],desired_pos[2]+Z_OFFSET_WYPT2+Z_OFFSET_WYPT3])
            else:
                waypt3 = np.array([desired_pos[0],desired_pos[1],desired_pos[2]+Z_OFFSET_WYPT2+Z_OFFSET_WYPT3+ANOTHER_OFFSET])
            desired_angle = self.set_desired_joint_angle(waypt3,PSI,gripper_change = False)

            ## SETS THE NEUTRAL POSITION FIVE OF THE ARM FOR PICK/PLACE AGAIN ##
            self.set_neutral_joint_angle(desired_angle)
            self.next_state = "execute"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self, pickNext = True, ARM_SPEED = 0.7):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """

        self.status_message = "State: Execute - Executing motion plan"
        if self.current_state == "pcknplce":
            self.next_state = "pcknplce"
        else:
            self.next_state = "idle"
        if pickNext == False:
            self.next_state = "idle"
        self.next_state = "mousemode"
        
        # Set the move time and accel time to proper values
        velocity_cap = 50.0 * np.pi/180.0
        default_accel_time = 1
        move_times = []
        accel_times = []
        for i in range(len(self.saved_waypoints)):
            displacement = 0
            if i == 0:
                displacement = self.rxarm.get_positions() - self.saved_waypoints[0]
            else:
                displacement = self.saved_waypoints[i-1] - self.saved_waypoints[i]
            
            displacement = max(abs(displacement))
            
            move_time = abs(displacement)/velocity_cap

            # Checks if the arm moves too fast (TUNABLE)
            if move_time < ARM_SPEED:
                move_time = ARM_SPEED
            
            if move_time >= default_accel_time*2:
                accel_time = default_accel_time
            else:
                accel_time = move_time/2
            move_times.append(move_time)
            accel_times.append(accel_time)

        assert(len(self.saved_waypoints) == len(self.saved_gripper_pos))
        prev_gripper = self.gripper_closed
        for (waypoint,gripper,move_time,accel_time) in zip(self.saved_waypoints, self.saved_gripper_pos,move_times,accel_times):
            # Stops the motion and moves it to the idle state
            if self.next_state == "stop_motion":
                break

            self.rxarm.set_joint_positions(waypoint, moving_time=move_time, accel_time=accel_time,blocking=False)

            rospy.sleep(move_time)

            # Check the gripper position and update it if necessary
            if gripper != prev_gripper:
                #TODO: wait for gripper
                if gripper:
                    self.rxarm.close_gripper()
                    self.gripper_closed = True
                else:
                    self.rxarm.open_gripper()
                    self.gripper_closed = False
                rospy.sleep(0.2)
            prev_gripper = gripper

    # Kind of a useless state? Can save/clear waypoints when not in teach mode
    def teach(self):
        if self.saved_waypoints is not None:
            if len(self.saved_waypoints) > 0:
                self.status_message = str(self.saved_waypoints[-1])
            else:
                self.status_message = "State: Teach -- Executing teach"
        else:
            self.status_message = "State: Teach -- Executing teach"

    def save_waypoint(self):
        self.status_message = "Saving waypoint"
        pos = self.rxarm.get_positions()
        if self.saved_waypoints is not None:
            self.saved_waypoints = np.vstack((self.saved_waypoints, pos))
        else:
            self.saved_waypoints = pos
        if self.saved_gripper_pos is not None:
            self.saved_gripper_pos = np.append(self.saved_gripper_pos, self.gripper_closed)
        else:
            self.saved_gripper_pos = self.gripper_closed
        self.status_message = str(pos)
        self.next_state = "teach"
    
    def clear_previous_waypoint(self):
        self.saved_waypoints.pop()
        self.saved_gripper_pos.pop()
        self.status_message = "State: Last waypoint cleared"
        self.next_state = "teach"

    def print_waypoints(self):
        print(str(self.saved_waypoints))
        print(str(self.saved_gripper_pos))
        self.next_state = "teach"

    # If the arm is in motion in the execute state, it will stop the motion of the arm
    # without setting torque of the arm to zero
    def stop_motion(self):
        self.status_message = "State: Motion stopped"
        if self.prev_state == "pcknplce":
            self.next_state = "mousemode"
        else:
            self.next_state = "idle"

    def threshold(self, coord, min_thresh, max_thresh):
        "Returns boolean whether it is in valid threshold"
        return coord > min_thresh and coord < max_thresh

    def sort_tuple(self, tup):
        """Puts smaller blocks ahead of larger blocks and those closer to the arm."""
        #tup = sorted(tup, key = lambda x: (x[3], x[1]), reverse=True)
        tup = sorted(tup, key = lambda x: (x[3]=="True", float(x[1])))
        return tup

    def sort_tuple_z(self, tup):
        """Puts higher blocks ahead of lower blocks always."""
        tup = sorted(tup, key = lambda x: float(x[2]), reverse=True)
        return tup

    def sort_tuple_color(self, tup):
        """Puts blocks in color order, first small blocks then big blocks"""
        tup = sorted(tup, key = lambda x: (x[3]=="False", float(x[4])))
        return tup

    def pick_n_sort(self):
        "Drop small blocks to the left and big ones to the right of the arm in the negative half-plane."
        self.prev_state = "pick_n_sort"
        print("IN PICK N SORT")
        self.PSI = np.pi/4
        X_THRESH_MIN = -500 #Whats the lowest x it should detect
        X_THRESH_MAX = 500 #Highest x 
        Y_THRESH_MIN = 0
        Y_THRESH_MAX = 500

        X_BASE_MIN = -100 #Whats the lowest x it should detect
        X_BASE_MAX = 100 #Highest x 
        Y_BASE_MIN = -175
        Y_BASE_MAX = 75
        
        LARGE_BLOCK = 65
        SMALL_BLOCK = 50
        x_offset_big = 0
        x_offset_small = 0
        y_offset_big = 0
        y_offset_small = 0
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, Y_THRESH_MAX)):
                if (self.threshold(float(block[0]), X_BASE_MIN, X_BASE_MAX) and self.threshold(float(block[1]), Y_BASE_MIN, Y_BASE_MAX)):
                    continue
                self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])
                self.next_state = "picknplace"
                self.pickAndPlace(False)
                if self.next_state == "stop_motion":
                    self.next_state = "idle"
                    break
                self.execute()
                if block[3] == "True":
                    self.desired_pose = np.array([400-x_offset_big, -100+y_offset_big, 0])
                    self.large_block = True
                    x_offset_big += LARGE_BLOCK
                    if (x_offset_big > 350):
                        x_offset_big = 0
                        y_offset_big += LARGE_BLOCK
                else:
                    self.desired_pose = np.array([-400+x_offset_small, -75+y_offset_small, 0])
                    self.large_block = False
                    x_offset_small += SMALL_BLOCK
                    if (x_offset_small > 300):
                        x_offset_small = 0
                        y_offset_small += SMALL_BLOCK
                self.pickAndPlace(False)
                if self.next_state == "stop_motion":
                    self.next_state = "idle"
                    break
                self.execute()
            else:
                continue

        again = False
        self.detect()
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, Y_THRESH_MAX)):
                again = True
                break
        if again:
            self.next_state = "pick_n_sort"
        else:
            self.next_state = "idle"

    def pick_n_stack(self):
        "Stack all blocks three tall to the left or right of the arm in the negative half-plane"
        self.prev_state = "pick_n_stack"
        print("IN PICK N STACK")
        self.PSI = np.pi/4
        X_THRESH_MIN = -475 #Whats the lowest x it should detect
        X_THRESH_MAX = 475 #Highest x 
        Y_THRESH_MIN = 0
        Y_THRESH_MAX = 500

        X_BASE_MIN = -100 
        X_BASE_MAX = 100
        Y_BASE_MIN = -175
        Y_BASE_MAX = 75
        
        again = False

        if again == False:
            x_offset_big = 0
            y_offset_big = 0
            x_offset_small = 0
            y_offset_small = 0

        self.detected_blocks = self.sort_tuple(self.detected_blocks)

        #1. Pull all small blocks in the LHP first (non stacked)
        self.large_block = False
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, Y_THRESH_MAX)):
                if (self.threshold(float(block[0]), X_BASE_MIN, X_BASE_MAX) and self.threshold(float(block[1]), Y_BASE_MIN, Y_BASE_MAX)):
                    continue
                if(block[3] == "False"):
                    self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])
                    self.next_state = "picknplace"
                    self.pickAndPlace(False)
                    if self.next_state == "stop_motion":
                        self.next_state = "idle"
                        break
                    self.execute()
                    self.desired_pose = np.array([-300+x_offset_small, -75+y_offset_small, 0])
                    x_offset_small += 75
                    if (x_offset_small > 200):
                        x_offset_small = 0
                        y_offset_small += 50
                    self.pickAndPlace(False)
                    if self.next_state == "stop_motion":
                        self.next_state = "idle"
                        break
                    self.execute()
            else:
                continue

        self.detect()
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, Y_THRESH_MAX)):
                again = True
                break
        if again:
            self.current_state = "idle"
            self.current_state = "pick_n_stack"

        #2. Pick the big blocks and stack them until three high
        self.detect()
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        self.PSI = np.pi/4
        counter = 0
        self.large_block = True
        prev_z = 0
        curr_z = 0
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, Y_THRESH_MAX)):
                if (self.threshold(float(block[0]), X_BASE_MIN, X_BASE_MAX) and self.threshold(float(block[1]), Y_BASE_MIN, Y_BASE_MAX)):
                    continue
                self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])
                self.next_state = "picknplace"
                self.pickAndPlace(False)
                if self.next_state == "stop_motion":
                    self.next_state = "idle"
                    break
                self.execute()
                self.desired_pose = np.array([400 - x_offset_big, -100+ y_offset_big, curr_z])
                prev_z = float(block[2])
                curr_z += prev_z
                if counter >= 2:
                    x_offset_big += 100
                    prev_z = 0
                    curr_z = 0
                    counter = 0
                if x_offset_big > 250:
                    x_offset_big = 0
                    y_offset_big = 75
                self.pickAndPlace(False)
                if self.next_state == "stop_motion":
                    self.next_state = "idle"
                    break
                self.execute()
                counter += 1
            else:
                continue
        
        self.detect()
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        #3. Stack the small blocks from the LHP on the unstacked large block
        self.large_block = False
        self.PSI = np.pi/4
        print("STARTing LHP detection")
        if(counter == 1):
            counter = 0
        for block in self.detected_blocks:
            # Checks for small blocks in the Bottom LHP
            if (self.threshold(float(block[0]), X_THRESH_MIN, 0) and self.threshold(float(block[1]), -200, 0)):
                if (self.threshold(float(block[0]), X_BASE_MIN, X_BASE_MAX) and self.threshold(float(block[1]), Y_BASE_MIN, Y_BASE_MAX)):
                    continue
                #if block[3] == "False":
                if True:
                    self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])
                    self.next_state = "picknplace"
                    self.pickAndPlace(False)
                    if self.next_state == "stop_motion":
                        self.next_state = "idle"
                        break
                    self.execute()
                    self.desired_pose = np.array([400 - x_offset_big, -100+y_offset_big, curr_z])
                    prev_z = float(block[2])
                    curr_z += prev_z
                    if counter >= 2:
                        x_offset_big += 100
                        prev_z = 0
                        curr_z = 0
                        counter = 0
                    if x_offset_big > 250:
                        x_offset_big = 0
                        y_offset_big = 75
                    self.pickAndPlace(False)
                    if self.next_state == "stop_motion":
                        self.next_state = "idle"
                        break
                    self.execute()
                    counter += 1
            else:
                continue

        #4. Pick and place the blocks in right negative half-plane
        self.detect()
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, Y_THRESH_MAX)):
                again = True
                break
        if again:
            self.next_state = "pick_n_stack"
        else:
            self.next_state = "idle"
            self.PSI = 0

    def block_near(self, x, y):
        """Returns True if there's a block within 75mm of the given x and y. False otherwise"""
        X_BASE_MIN = -100 
        X_BASE_MAX = 100
        Y_BASE_MIN = -175
        Y_BASE_MAX = 75
        if self.threshold(x, X_BASE_MIN, X_BASE_MAX) and self.threshold(y, Y_BASE_MIN, Y_BASE_MAX):
            #print("block_near:", x, ",", y, "WITHIN BASE")
            return True
        for block in self.detected_blocks:
            #print("block_near: Checking", block[3], "block at(", block[0], ",", block[1], ").")
            diff = ((float(block[0]) - x)**2 + (float(block[1]) - y)**2)**0.5
            if diff < 50:
                #print("block_near:", x, ",", y, "TOO CLOSE TO BLOCK AT", block[0], ",", block[1])
                return True
        #print("block_near:", x, ",", y, "CLEAR")
        return False

    def getNHPspot(self, x, y):
        """Returns new X and Y coordinates for NHP sorting"""
        newX = x
        newY = y

        # Offsets
        xOff = 100
        yOff = 100

        # If block in target location find open spot, else just return target coords
        dropZoneClear = False
        while not dropZoneClear:
            if self.block_near(newX,newY):
                # If in left NHP
                if newX < 0:
                    # Subtract offset from X
                    newX = newX - xOff
                    # If now out of range
                    if newX < -350:
                        # Start new row
                        newX = -125
                        newY = newY + yOff
                        if newY > 150:
                            newY = -100
                # If in right NHP
                else:
                    # Add offset to X
                    newX = newX + xOff
                    # If now out of range
                    if newX > 350:
                        # Start new row
                        newX = 125
                        newY = newY + yOff
                        if newY > 150:
                            newY = -100
            else:
                dropZoneClear = True
        return newX, newY

    def final_pos(self, color):
        if color == "Red":
            return -250, 200
        elif color == "Orange":
            return -175, 200
        elif color == "Yellow":
            return -100, 200
        elif color == "Green":
            return -25, 200
        elif color == "Blue":
            return 50, 200
        else: #color == "Purple":
            return 125, 200

    def staged_pos(self, color):
        if color == "Red":
            return -250, 300
        elif color == "Orange":
            return -175, 300
        elif color == "Yellow":
            return -100, 300
        elif color == "Green":
            return -25, 300
        elif color == "Blue":
            return 50, 300
        else: #color == "Purple":
            return 125, 300

    def placeBigBlock(self, block):
        if block[4] == "Red":
            self.saved_waypoints = self.bigR_waypoints
        elif block[4] == "Orange":
            self.saved_waypoints = self.bigO_waypoints
        elif block[4] == "Yellow":
            self.saved_waypoints = self.bigY_waypoints
        elif block[4] == "Green":
            self.saved_waypoints = self.bigG_waypoints
        elif block[4] == "Blue":
            self.saved_waypoints = self.bigB_waypoints
        else: #block[4] == "Purple":
            self.saved_waypoints = self.bigV_waypoints
        self.saved_gripper_pos = self.place_gripper_pos

        self.execute(False)

        self.saved_waypoints = None
        self.saved_gripper_pos = None

    def placeSmallBlock(self, block):
        if block[4] == "Red":
            self.saved_waypoints = self.smallR_waypoints
        elif block[4] == "Orange":
            self.saved_waypoints = self.smallO_waypoints
        elif block[4] == "Yellow":
            self.saved_waypoints = self.smallY_waypoints
        elif block[4] == "Green":
            self.saved_waypoints = self.smallG_waypoints
        elif block[4] == "Blue":
            self.saved_waypoints = self.smallB_waypoints
        else: #block[4] == "Purple":
            self.saved_waypoints = self.smallV_waypoints
        self.saved_gripper_pos = self.place_gripper_pos

        self.execute(False)

        self.saved_waypoints = None
        self.saved_gripper_pos = None

    def isOutOfBounds(self, block):
        X_THRESH_MIN = -450 #Whats the lowest x it should detect
        X_THRESH_MAX = 450 #Highest x 
        Y_THRESH_MIN = -150
        Y_THRESH_MAX = 450
        if float(block[0]) < X_THRESH_MIN:
            return True
        elif float(block[0]) > X_THRESH_MAX:
            return True
        elif float(block[1]) < Y_THRESH_MIN:
            return True
        elif float(block[1]) > Y_THRESH_MAX:
            return True
        else:
            return False

    
    def line_up(self):
        "Line up the large and small blocks in two separate lines in rainbow color order."
        self.PSI = 0
        # State variables
        liningUp = True
        stagingBlocks = True
        liningBig = True
        liningSmall = True
        slideBlocks = True

        # Default half-plane positions
        leftNHPspotX = -125
        leftNHPspotY = -100
        rightNHPspotX = 125
        rightNHPspotY = -100

        # Offset (see NHP point finder)
        xOff = 100
        yOff = 100

        # Height for blocks by default
        bazeZvalueBig = 0
        bazeZvalueSmall = 0

        # Manual z offset for stacked small blocks
        manZoff = 0

        # Colors to place, in order
        colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple"]

        # Loop until lines complete
        while liningUp:

            self.detect()

            # Table of staged
            stagedBlocks = np.vstack(([0,0,""],[0,0,""])) # Dummy blocks because I'm bad at python and because nothing else should ever be here

            ### STEP 1 - Determine if there are any stacks and unstack them ###
            while stagingBlocks:

                tries = 0
            
                # Table of blocks adjusted this loop
                currBlocks = np.vstack(([0,0],[0,0])) # Dummy blocks because I'm bad at python and because nothing else should ever be here
                
                # Sort blocks in descending z order
                sorted_detected_blocks = self.sort_tuple_z(self.detected_blocks)
                
                for block in sorted_detected_blocks: #staticBlockLocations:

                    # Reject out-of-bounds blocks
                    if self.isOutOfBounds(block):
                        print("REJECTING")
                        continue

                    # Avoid grabbing a staged block
                    staged = False
                    for stagedBlock in stagedBlocks:
                        if (((float(block[0]) - float(stagedBlock[0]))**2 + (float(block[1]) - float(stagedBlock[1]))**2)**0.5) < 99:
                            if (block[3] == "True" and float(block[0]) < 0) or (block[3] == "False" and float(block[0]) > 0):
                                staged = True
                                break
                    
                    # Avoid double-grabbing, particularly in the case of a small block on top of a big block
                    interferance = False
                    for currBlock in currBlocks:
                        if (((float(block[0]) - currBlock[0])**2 + (float(block[1]) - currBlock[1])**2)**0.5) < 50:
                            interferance = True
                            break

                    # See if block is covered by another
                    covered = False
                    for testBlock in sorted_detected_blocks:
                        if ((((float(block[0]) - float(testBlock[0]))**2 + (float(block[1]) - float(testBlock[1]))**2)**0.5) < 50) and (float(block[2]) <= float(testBlock[2])) and not (all(elem in block  for elem in testBlock)):
                            covered = True
                            break
                    
                    # See if block is the same as another
                    same = False
                    for testBlock in sorted_detected_blocks:
                        if (((float(block[0]) - float(testBlock[0]))**2 + (float(block[1]) - float(testBlock[1]))**2)**0.5) < 50 and not (all(elem in block  for elem in testBlock)):
                            same = True
                            break
                    
                    if covered or interferance or staged:
                        continue

                    # Save block coordinates
                    newBlock = [float(block[0]), float(block[1])]
                    currBlocks = np.vstack((currBlocks, newBlock))

                    # If large block
                    if (block[3] == "True"):
                        print("Detected large", block[4], "block at (", block[0], ",", block[1], ",", block[2], ").")

                        # Move to desired spot

                        # Is large block
                        self.large_block = True

                        # Set block position
                        self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])

                        # Pick up block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Determine if block can be moved to final position or if it must be staged
                        finX, finY = self.final_pos(block[4])
                        stagX, stagY = self.staged_pos(block[4])
                        # Get NHP spot coordinates and set as desired position
                        leftNHPspotX, leftNHPspotY = self.getNHPspot(leftNHPspotX, leftNHPspotY)
                        self.desired_pose = np.array([leftNHPspotX, leftNHPspotY, bazeZvalueBig])

                        print("Moving it to a clear spot at (", self.desired_pose[0], ",", self.desired_pose[1], ").")

                        # Place block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Adjust NHP coordinates
                        leftNHPspotX = leftNHPspotX - xOff
                        # If now out of range
                        if leftNHPspotX < -350:
                            # Start new row
                            leftNHPspotX = -125
                            leftNHPspotY = leftNHPspotY + yOff
                            if leftNHPspotY > 150:
                                leftNHPspotY = -100

                        # Save block coordinates as staged
                        movedBlock = [self.desired_pose[0], self.desired_pose[1], block[4]]
                        if movedBlock[1] < 150:
                            stagedBlocks = np.vstack((stagedBlocks, movedBlock))

                    elif (block[3] == "False"):
                        print("Detected small", block[4], "block at (", block[0], ",", block[1], ",", block[2], ").")
                        # Move to right half plane

                        # Is small block
                        self.large_block = False

                        # Set block position
                        self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2]) + manZoff])

                        # Pick up block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Get NHP spot coordinates and set as desired position
                        rightNHPspotX, rightNHPspotY = self.getNHPspot(rightNHPspotX, rightNHPspotY)
                        self.desired_pose = np.array([rightNHPspotX, rightNHPspotY, bazeZvalueSmall])

                        print("Moving it to a clear spot at (", self.desired_pose[0], ", ", self.desired_pose[1], ").")

                        # Place block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Adjust NHP coordinates
                        rightNHPspotX = rightNHPspotX + xOff
                        # If now out of range
                        if rightNHPspotX > 350:
                            # Start new row
                            rightNHPspotX = 125
                            rightNHPspotY = rightNHPspotY + yOff
                            if rightNHPspotY > 150:
                                rightNHPspotY = -100

                        # Save block coordinates as staged
                        movedBlock = [self.desired_pose[0], self.desired_pose[1], block[4]]
                        if movedBlock[1] < 150:
                            stagedBlocks = np.vstack((stagedBlocks, movedBlock))

                    # Refresh detections (may cause the same blocks to be re-analyzed for stackage, idk, if so that's fine)
                    self.detect()

                # Refresh detections (may cause the same blocks to be re-analyzed for stackage, idk, if so that's fine)
                self.detect()

                # Break loop if all blocks staged
                allStaged = True
                colorFoundLHP = [False, False, False, False, False, False]
                colorFoundRHP = [False, False, False, False, False, False]
                for block in self.detected_blocks:
                    thisStaged = False
                    for stagedBlock in stagedBlocks:
                        diff = ((float(block[0]) - float(stagedBlock[0]))**2 + (float(block[1]) - float(stagedBlock[1]))**2)**0.5
                        if diff < 100 and stagedBlock[2] == block[4]:
                            thisStaged = True
                            if float(block[0]) < 0:
                                for i in range(len(colorFoundLHP)):
                                    if colors[i] == block[4]:
                                        colorFoundLHP[i] = True
                                        break
                            else:
                                for i in range(len(colorFoundRHP)):
                                    if colors[i] == block[4]:
                                        colorFoundRHP[i] = True
                    if not thisStaged:
                        allStaged = False

                if allStaged and all(colorFoundLHP) and (all(colorFoundRHP) or (not any(colorFoundRHP))):
                    allStaged = True
                else:
                    allStaged = False

                # Move on only when all blocks staged
                if allStaged:
                    stagingBlocks = False

            print("COMPLETED STAGING")
            
            # Put the big blocks in a line
            while liningBig == True:

                stagedCount = 0
                
                allColorsNotFound = True
                while allColorsNotFound:
                    allColorsNotFound = False
                    print("Starting loop")
                    for color in colors:
                        self.detect()
                        print("Trying to identify", color)
                        colorFound = False
                        for block in self.detected_blocks:
                            if ((float(block[0]) < 0 and float(block[1]) < 150) or (float(block[1]) > 150)) and (block[4] == color):# and (block[3] == "True"):
                                print("Matched", color)
                                colorFound = True
                                if not float(block[1])>150:
                                    print("Moving block")
                                    self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])

                                    # Pick up block
                                    self.pickAndPlace(False)
                                    if self.next_state == "stop_motion":
                                        self.next_state = "idle"
                                        liningUp = False
                                        break
                                    self.execute()

                                    # Place block in dedicated spot
                                    self.placeBigBlock(block)

                                    # Skip any duplicates of this color
                                    break
                        if not colorFound:
                            print("Color", color, "not found!")
                            allColorsNotFound = True
                            break
                        

                # This should prolly just be a function or something don't @ me
                liningBig = False

            while liningSmall == True:
                
                allColorsNotFound = True
                while allColorsNotFound:
                    allColorsNotFound = False
                    print("Starting loop")
                    for color in colors:
                        self.detect()
                        print("Trying to identify", color)
                        colorFound = False
                        for block in self.detected_blocks:
                            if ((float(block[0]) > 0 and float(block[1]) < 150) or (float(block[1]) > 225)) and (block[4] == color):# and (block[3] == "False"):
                                print("Matched", color)
                                colorFound = True
                                if not float(block[1])>150:
                                    print("Moving block")
                                    self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])

                                    # Pick up block
                                    self.pickAndPlace(False)
                                    if self.next_state == "stop_motion":
                                        self.next_state = "idle"
                                        liningUp = False
                                        break
                                    self.execute()

                                    # Place block in dedicated spot
                                    self.placeSmallBlock(block)

                                    # Skip any duplicates of this color
                                    break
                        if not colorFound:
                            print("Color", color, "not found!")
                            allColorsNotFound = True
                            break
                
                # This should prolly just be a function or something don't @ me
                liningSmall = False

            while slideBlocks == True:

                # Slide them together at the end for wow factor and literally no other reason lmfao
                
                # Slide the big blocks together
                self.saved_waypoints = self.bigSlide_waypoints
                self.saved_gripper_pos = self.slide_gripper_pos
                self.execute(False, 2)
                self.saved_waypoints = None
                self.saved_gripper_pos = None

                # Slide the small blocks together
                self.saved_waypoints = self.smallSlide_waypoints
                self.saved_gripper_pos = self.slide_gripper_pos
                self.execute(False, 2)
                self.saved_waypoints = None
                self.saved_gripper_pos = None

                # Reset the arm
                self.saved_waypoints = self.reset_waypoints
                self.saved_gripper_pos = self.reset_gripper_pos
                self.execute(False)
                self.saved_waypoints = None
                self.saved_gripper_pos = None

                # This should prolly just be a function or something don't @ me
                slideBlocks = False

            # Temporary killswitch
            liningUp = False # Now permanent lmao
            print("COMPLETED")
                
        self.next_state = "idle"

    def stack_rainbow(self):
        """"Small and big blocks have been sorted"""
        print("STACK RAINBOW")
        self.PSI = 0 #np.pi/4
        X_THRESH_MIN = -500 #Whats the lowest x it should detect
        X_THRESH_MAX = 500 #Highest x 
        Y_THRESH_MIN = -250

        X_BASE_MIN = -100 #Whats the lowest x it should detect
        X_BASE_MAX = 100 #Highest x 
        Y_BASE_MIN = -175
        Y_BASE_MAX = 75
        
        SMALL_BLOCK = 25
        extra_offset = 10
        large_block_counter = 0
        
        for block in self.detected_blocks:
            if block[4] == "Red":
                block[4] = 0
            if block[4] == "Orange":
                block[4] = 1
            if block[4] == "Yellow":
                block[4] = 2
            if block[4] == "Green":
                block[4] = 3
            if block[4] == "Blue":
                block[4] = 4
            if block[4] == "Purple":
                block[4] = 5
        self.PSI = np.pi/4
        self.detected_blocks = self.sort_tuple_color(self.detected_blocks)
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, 150)):
                #Only detect blocks in the NHP
                if (self.threshold(float(block[0]), X_BASE_MIN, X_BASE_MAX) and self.threshold(float(block[1]), Y_BASE_MIN, Y_BASE_MAX)):
                    continue
                self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])
                self.next_state = "picknplace"
                self.pickAndPlace(False)
                if self.next_state == "stop_motion":
                    self.next_state = "idle"
                    break
                self.execute()
                if float(block[0]) < 0:
                    self.desired_pose = np.array([-200, 275, self.z_offset_large])
                    self.large_block = True
                    self.z_offset_large += float(block[2]) + extra_offset + 3*large_block_counter
                    large_block_counter += 1
                else:
                    self.desired_pose = np.array([150, 375, self.z_offset_small])
                    self.large_block = False
                    self.z_offset_small += SMALL_BLOCK
                self.pickAndPlace(False)
                if self.next_state == "stop_motion":
                    self.next_state = "idle"
                    break
                self.execute()
            else:
                continue

        again = False
        self.detect()
        self.detected_blocks = self.sort_tuple(self.detected_blocks)
        for block in self.detected_blocks:
            if (self.threshold(float(block[0]), X_THRESH_MIN, X_THRESH_MAX) and self.threshold(float(block[1]), Y_THRESH_MIN, 150)):
                again = True
                break
        if again:
            self.next_state = "stack_rainbow"
        else:
            self.next_state = "idle"
    
    def stack_high_new(self):
        "Stage the blocks for stack high."

        self.PSI = np.pi/4
        # State variables
        liningUp = False #Change to True if we want to stage
        stagingBlocks = True

        # Default half-plane positions
        leftNHPspotX = -125
        leftNHPspotY = -100
        rightNHPspotX = 125
        rightNHPspotY = -100

        # Offset (see NHP point finder)
        xOff = 100
        yOff = 100

        # Height for blocks by default
        bazeZvalueBig = 0
        bazeZvalueSmall = 0

        # Manual z offset for stacked small blocks
        manZoff = 0

        # Colors to place, in order
        colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple"]

        # Loop until lines complete
        while liningUp:

            self.detect()

            # Table of staged
            stagedBlocks = np.vstack(([0,0,""],[0,0,""])) # Dummy blocks because I'm bad at python and because nothing else should ever be here

            ### STEP 1 - Determine if there are any stacks and unstack them ###
            while stagingBlocks:

                # Table of blocks adjusted this loop
                currBlocks = np.vstack(([0,0],[0,0])) # Dummy blocks because I'm bad at python and because nothing else should ever be here
                
                # Sort blocks in descending z order
                sorted_detected_blocks = self.sort_tuple_z(self.detected_blocks)
                
                for block in sorted_detected_blocks: #staticBlockLocations:

                    # Reject out-of-bounds blocks
                    if self.isOutOfBounds(block):
                        print("REJECTING")
                        continue

                    # Avoid grabbing a staged block
                    staged = False
                    for stagedBlock in stagedBlocks:
                        if (((float(block[0]) - float(stagedBlock[0]))**2 + (float(block[1]) - float(stagedBlock[1]))**2)**0.5) < 99:
                            if (block[3] == "True" and float(block[0]) < 0) or (block[3] == "False" and float(block[0]) > 0):
                                staged = True
                                break
                    
                    # Avoid double-grabbing, particularly in the case of a small block on top of a big block
                    interferance = False
                    for currBlock in currBlocks:
                        if (((float(block[0]) - currBlock[0])**2 + (float(block[1]) - currBlock[1])**2)**0.5) < 50:
                            interferance = True
                            break

                    # See if block is covered by another
                    covered = False
                    for testBlock in sorted_detected_blocks:
                        if ((((float(block[0]) - float(testBlock[0]))**2 + (float(block[1]) - float(testBlock[1]))**2)**0.5) < 50) and (float(block[2]) <= float(testBlock[2])) and not (all(elem in block  for elem in testBlock)):
                            covered = True
                            break
                    
                    # See if block is the same as another
                    same = False
                    for testBlock in sorted_detected_blocks:
                        if (((float(block[0]) - float(testBlock[0]))**2 + (float(block[1]) - float(testBlock[1]))**2)**0.5) < 50 and not (all(elem in block  for elem in testBlock)):
                            same = True
                            break
                    
                    if covered or interferance or staged:
                        continue

                    # Save block coordinates
                    newBlock = [float(block[0]), float(block[1])]
                    currBlocks = np.vstack((currBlocks, newBlock))

                    # If large block
                    if (block[3] == "True"):
                        print("Detected large", block[4], "block at (", block[0], ",", block[1], ",", block[2], ").")

                        # Move to desired spot

                        # Is large block
                        self.large_block = True

                        # Set block position
                        self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2])])

                        # Pick up block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Determine if block can be moved to final position or if it must be staged
                        finX, finY = self.final_pos(block[4])
                        stagX, stagY = self.staged_pos(block[4])
                        leftNHPspotX, leftNHPspotY = self.getNHPspot(leftNHPspotX, leftNHPspotY)
                        self.desired_pose = np.array([leftNHPspotX, leftNHPspotY, bazeZvalueBig])

                        print("Moving it to a clear spot at (", self.desired_pose[0], ",", self.desired_pose[1], ").")
                        # Place block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Adjust NHP coordinates
                        leftNHPspotX = leftNHPspotX - xOff
                        # If now out of range
                        if leftNHPspotX < -350:
                            # Start new row
                            leftNHPspotX = -125
                            leftNHPspotY = leftNHPspotY + yOff
                            if leftNHPspotY > 150:
                                leftNHPspotY = -100

                        # Save block coordinates as staged
                        movedBlock = [self.desired_pose[0], self.desired_pose[1], block[4]]
                        if movedBlock[1] < 150:
                            stagedBlocks = np.vstack((stagedBlocks, movedBlock))

                    elif (block[3] == "False"):
                        print("Detected small", block[4], "block at (", block[0], ",", block[1], ",", block[2], ").")

                        # Move to right half plane

                        # Is small block
                        self.large_block = False

                        # Set block position
                        self.desired_pose = np.array([float(block[0]), float(block[1]), float(block[2]) + manZoff])

                        # Pick up block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Get NHP spot coordinates and set as desired position
                        rightNHPspotX, rightNHPspotY = self.getNHPspot(rightNHPspotX, rightNHPspotY)
                        self.desired_pose = np.array([rightNHPspotX, rightNHPspotY, bazeZvalueSmall])

                        print("Moving it to a clear spot at (", self.desired_pose[0], ", ", self.desired_pose[1], ").")

                        # Place block
                        self.pickAndPlace(False)
                        if self.next_state == "stop_motion":
                            self.next_state = "idle"
                            liningUp = False
                            break
                        self.execute()

                        # Adjust NHP coordinates
                        rightNHPspotX = rightNHPspotX + xOff
                        # If now out of range
                        if rightNHPspotX > 350:
                            # Start new row
                            rightNHPspotX = 125
                            rightNHPspotY = rightNHPspotY + yOff
                            if rightNHPspotY > 150:
                                rightNHPspotY = -100

                        # Save block coordinates as staged
                        movedBlock = [self.desired_pose[0], self.desired_pose[1], block[4]]
                        if movedBlock[1] < 150:
                            stagedBlocks = np.vstack((stagedBlocks, movedBlock))

                    # Refresh detections (may cause the same blocks to be re-analyzed for stackage, idk, if so that's fine)
                    self.detect()

                # Refresh detections (may cause the same blocks to be re-analyzed for stackage, idk, if so that's fine)
                self.detect()

                # Break loop if all blocks staged
                allStaged = True
                colorFoundLHP = [False, False, False, False, False, False]
                colorFoundRHP = [False, False, False, False, False, False]
                for block in self.detected_blocks:
                    thisStaged = False
                    for stagedBlock in stagedBlocks:
                        diff = ((float(block[0]) - float(stagedBlock[0]))**2 + (float(block[1]) - float(stagedBlock[1]))**2)**0.5
                        if diff < 100 and stagedBlock[2] == block[4]:
                            thisStaged = True
                            if float(block[0]) < 0:
                                for i in range(len(colorFoundLHP)):
                                    if colors[i] == block[4]:
                                        colorFoundLHP[i] = True
                                        break
                            else:
                                for i in range(len(colorFoundRHP)):
                                    if colors[i] == block[4]:
                                        colorFoundRHP[i] = True
                    if not thisStaged:
                        allStaged = False

                if allStaged and all(colorFoundLHP) and (all(colorFoundRHP) or (not any(colorFoundRHP))):
                    allStaged = True
                else:
                    allStaged = False

                # Move on only when all blocks staged
                #Uncomment following for competition
                if allStaged:
                    stagingBlocks = False

            print("COMPLETED STAGING")
            liningUp = False

        self.next_state = "stack_rainbow"

    def use_apriltag(self, msg):
        
        # Initialize empty arrays to store april tag location/orientation/tag_IDs
        positions = np.array([])
        orientations = np.array([])
        tag_ids = np.array([])
        
        # Stores the quaternion and position of each apriltag from the detections message
        for tag in msg.detections:
            tag_id = tag.id
            z_pos = tag.pose.pose.pose.position.z
            x_pos = tag.pose.pose.pose.position.x
            y_pos = tag.pose.pose.pose.position.y
            x_ori = tag.pose.pose.pose.orientation.x
            y_ori = tag.pose.pose.pose.orientation.y
            z_ori = tag.pose.pose.pose.orientation.z
            w_ori = tag.pose.pose.pose.orientation.w
            positions = np.append(positions,np.array([x_pos, y_pos, z_pos]))
            orientations = np.append(orientations,np.array([x_ori, y_ori, z_ori, w_ori]))
            tag_ids = np.append(tag_ids, tag_id)
        
        # Confirm numPoints matches the number of tags seen
        assert(tag_ids.shape[0] >= self.numPoints)
        for i in range(self.numPoints):
            tag_exists = False
            for tag_id in tag_ids:
                if tag_id == i+1:
                    tag_exists = True
            assert(tag_exists)
        
        # Reshape arrays
        self.src_pts_pos = np.reshape(positions,(tag_ids.shape[0],3))
        self.src_pts_ori = np.reshape(orientations,(tag_ids.shape[0],4))

        # Sorts the output of the apriltag positions based on ID value. 
        sort_indices = np.argsort(tag_ids)
        self.src_pts_pos = self.src_pts_pos[sort_indices]
        self.tag_ids = tag_ids[sort_indices]

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """Perform camera calibration routine here"""

        # Create a ROS object that subscribes to /tag_detections message and calls the use_apriltag function
        # Unregisters after 1 second to only take one instance of apriltag XYZ instead of a continuous stream
        april_tag = rospy.Subscriber('/tag_detections', String, self.use_apriltag)
        rospy.sleep(1)
        april_tag.unregister()
        
        # Extrinsic Camera Calibration
        # Use the Apriltag X,Y,Z and convert to camera coordinates
        # Gets image points from the camera from Xc, Yc, Zc into u, v
        image_points = []
        for pos in self.src_pts_pos:
            # Multiplies the Xc, Yc, Zc tag positions with the camera intrinsic matrix and divide by the depth
            # to return (u,v,1). See transformation lecture slides for more details
            image_points = np.append(image_points, np.matmul(self.camera_intrinsic,pos)/pos[2])
        image_points = np.reshape(image_points,(self.numPoints,3))

        # Removes the "z" column in (u,v,1) to make it only a 2D coordinate
        image_points = image_points[:,:-1]
        self.board_tag_image_points = image_points.astype(int)[:4, :2]
        image_points = np.ascontiguousarray(image_points[:,:2]).reshape((self.numPoints,1,2))
        
        APRIL_TAG_DEBUG_MODE = False
        if (APRIL_TAG_DEBUG_MODE):
            print("\nTag IDs and Positions:")
            print(self.tag_ids)
            print(self.src_pts_pos)
            print("\n")
            print("Board Tag Image Points")
            print(self.board_tag_image_points)
            print("\n")

        # Change parameters into float32 for pnp input arguments (see below) 
        image_points = image_points.astype(np.float32)
        self.camera_intrinsic = self.camera_intrinsic.astype(np.float32)
        self.distortion_coef = self.distortion_coef.astype(np.float32)

        # Select destination points to apply the homography transform to
        model_pts = self.april_tags.reshape((self.april_tags.shape[0]/3, 3))
        model_pts = model_pts[:self.numPoints]
        model_pts = model_pts.astype(np.float32)

        # Retrieve the camera extrinsic from the cv2.PnP function that takes the april tag locations
        self.camera_extrinsic = recover_homogenous_transform_pnp(model_pts, image_points, self.camera_intrinsic, self.distortion_coef)

        print("Completed Calibration")
        print("\n")
        self.calibrated = True
        self.status_message = "Calibration - Completed Calibration"

    def detect(self):
        """!
        @brief      Detect the blocks
        returns list of blocks(tuple) where each block(tuple) has following values:
        block[0]: x coordinate
        block[1]: y coordinate
        block[2]: z coordinate
        block[3]: Boolean (is it large?)
        block[4]: color (string)
        """
        if (self.camera.calibrated == True) and (self.camera.DepthFrameRaw.any() != 0) and (len(self.camera.block_detections) > 0): #CHANGE THIS TO THE BUTTON OR SOMETHING
            all_blocks = self.camera.block_detections
            pts = [(val, key) for color in all_blocks for key, vals in color.items() for val in vals]
            blocks_world_coordinates = np.array([])

            depth_image = deepcopy(self.camera.DepthFrameRaw)
            z_orig = cv2.warpPerspective(depth_image, self.camera.depth_transform_matrix, (depth_image.shape[1], depth_image.shape[0]))
            
            for pt,color in pts:
                pt_x, pt_y = pt[0]
                height,width  = pt[1]
                large = False
                if (height * width > 2200) or ((height > 45) or (width > 45)):
                    large = True
                pt = [pt_x, pt_y]
                pt[0] = int(pt[0])
                pt[1] = int(pt[1])
                z = self.camera.DepthFrameRaw[pt[1]][pt[0]].copy()

                # Get intrinsic and extrinsic matricies
                inM = self.camera_intrinsic.copy()
                exM = self.camera_extrinsic.copy()
                
                self.camera.intrinsic_matrix = inM
                self.camera.extrinsic_matrix = exM

                # Get mouse points [u,v,1]
                pt_np = np.array([pt[0], pt[1], 1])

                if (self.camera.homography_matrix is not None):
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

                    z = z_orig[pt_np[1]][pt_np[0]]
                # Multiply inverse of intrinsic matrix by mouse points and scale by z
                coord_c = np.matmul(np.linalg.inv(inM),pt_np)
                coord_c = coord_c * z

                # Convert to Homogeneous coordinates
                coord_c = np.append(coord_c, 1)

                # Multiply inverse of extrinsic matrix by Homogeneous coordinates to get world coordinates
                [block_x,block_y,block_z,_] = np.matmul(np.linalg.inv(exM),coord_c)

                # If block z height too small, skip it
                if block_z < 5:
                    continue

                # If block detected within base area, skip it
                if self.threshold(block_x, self.X_BASE_MIN, self.X_BASE_MAX) and self.threshold(block_y, self.Y_BASE_MIN, self.Y_BASE_MAX):
                    continue

                # Pass stuff to build block coordinate package
                block_world_coordinates = np.array([float(block_x), float(block_y), float(block_z), large, color])
                block_world_coordinates[0] = float(block_world_coordinates[0])
                block_world_coordinates[1] = float(block_world_coordinates[1])
                block_world_coordinates[2] = float(block_world_coordinates[2])
                if(np.size(blocks_world_coordinates) == 0):
                    blocks_world_coordinates = block_world_coordinates
                else:
                    blocks_world_coordinates = np.vstack((blocks_world_coordinates, block_world_coordinates))     
            
            self.detected_blocks = blocks_world_coordinates
            if self.detected_blocks.shape == (5,):
                self.detected_blocks = np.array([self.detected_blocks])

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)