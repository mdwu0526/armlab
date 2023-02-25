"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    Ai_s = []
    for link_prev in range(0,link+1):
        (a, alpha, d, theta_offset) = dh_params[link_prev]
        Ai = get_transform_from_dh(a, alpha, d, theta_offset + joint_angles[link_prev])
        Ai_s.append(Ai)
    final_H = Ai_s[-1]
    for index in range(len(Ai_s)-2, -1, -1):
        final_H = np.matmul(Ai_s[index],final_H,)
    return final_H


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    trans = np.zeros([4,4])
    trans[0,0] = np.cos(theta)
    trans[0,1] = -1 * np.sin(theta) * np.cos(alpha)
    trans[0,2] = np.sin(theta) * np.sin(alpha)
    trans[1,0] = np.sin(theta)
    trans[1,1] = np.cos(theta) * np.cos(alpha)
    trans[1,2] = -1 * np.cos(theta) * np.sin(alpha)
    trans[2,1] = np.sin(alpha)
    trans[2,2] = np.cos(alpha)
    trans[0,3] = a * np.cos(theta)
    trans[1,3] = a * np.sin(theta)
    trans[2,3] = d
    trans[3,3] = 1
    return trans


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """

    T = T[:-1, :-1]
    r = R.from_dcm(T)
    angles = r.as_euler('ZYZ')
    PHI = angles[2]
    THETA = angles[1]
    PSI = angles[0]

    return np.array([PHI, THETA, PSI])


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    t = T[:,-1]
    theta = []
    theta.append(np.stack(get_euler_angles_from_T(T)))
    theta = np.stack(theta)
    t = t[:-1]
    pose = np.concatenate([t,theta],axis=None)
    return list(pose)


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def IK_geometric(pose, PSI):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array (end effector pose)
    @param      PSI        The angle from the horizontal of the arm

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    # Robot parameters
    BASE = 103.91
    L1 = np.sqrt(50**2+200**2)
    L2 = 200
    L3 = 174.15
    ELBOW_ANGLE = np.arctan2(50,200)

    # Currently works only in front of the robot arm because of clamp()
    theta_1_1 = -1*np.arctan2(pose[0],pose[1]) # Front Orientation
    theta_1_2 = np.pi + np.arctan2(pose[0],pose[1]) # Backward Orientation
    
    # Checks if the current pose position in X, Y, and Z can be reached within the arm's span
    R05 = R.from_euler('ZYZ',[np.pi/2+theta_1_1, np.pi/2+PSI, -np.pi])
    R05 = R05.as_dcm() # Rotation from base frame to EE frame
    XYZ = np.array(pose[0:3])
    oWC = XYZ - L3*np.matmul(R05,np.array([0,0,1]).T) #Location of wrist center
    Xc = oWC[0]
    Yc = oWC[1]
    Zc = oWC[2]

    # Solve for theta angles
    r = np.sqrt(Xc**2+Yc**2)
    s = Zc-BASE

    # Angle of the elbow
    theta_3_1 = np.arccos((-L1**2-L2**2+(r**2+s**2)) / (2*L1*L2))# Elbow down
    theta_3_2 = -1*np.arccos((-L1**2-L2**2+(r**2+s**2)) /(2*L1*L2)) # Elbow up
    theta_3_3 = -1*np.arccos((-L1**2-L2**2+(r**2+s**2)) /(2*L1*L2)) # Elbow down rotated
    theta_3_4 = np.arccos((-L1**2-L2**2+(r**2+s**2)) / (2*L1*L2))# Elbow up rotated

    # Angle of the shoulder
    theta_2_1 = np.arctan2(s,r) - np.arctan2( L2*np.sin(theta_3_1) , L1+L2*np.cos(theta_3_1) ) # Elbow down
    theta_2_2 = np.arctan2(s,r) - np.arctan2( L2*np.sin(theta_3_2) , L1+L2*np.cos(theta_3_2) ) # Elbow up
    theta_2_3 = np.pi - np.arctan2(s,r) - np.arctan2( L2*np.sin(theta_3_3) , L1+L2*np.cos(theta_3_3) ) # Elbow down rotated
    theta_2_4 = np.pi - np.arctan2(s,r) - np.arctan2( L2*np.sin(theta_3_4) , L1+L2*np.cos(theta_3_4) ) # Elbow up rotated
    
    # Angle of the wrist
    theta_4_1 = -PSI-(theta_2_1+theta_3_1) # Elbow down
    theta_4_2 = -PSI-(theta_2_2+theta_3_2) # Elbow up
    theta_4_3 = PSI+np.pi-(theta_2_3+theta_3_3) # Elbow down rotated
    theta_4_4 = PSI+np.pi-(theta_2_4+theta_3_4) # Elbow up rotated
    
    # Shoulder Joint Angles
    shldJ_1 = np.pi/2-theta_2_1-ELBOW_ANGLE # Elbow down
    shldJ_2 = np.pi/2-theta_2_2-ELBOW_ANGLE # Elbow up
    shldJ_3 = np.pi/2-theta_2_3-ELBOW_ANGLE # Elbow down rotated
    shldJ_4 = np.pi/2-theta_2_4-ELBOW_ANGLE # Elbow up rotated

    # Elbow Joint Angles
    elbJ_1 = np.pi/2+theta_3_1-ELBOW_ANGLE # Elbow down
    elbJ_2 = np.pi/2+theta_3_2-ELBOW_ANGLE # Elbow up
    elbJ_3 = np.pi/2+theta_3_3-ELBOW_ANGLE # Elbow down rotated
    elbJ_4 = np.pi/2+theta_3_4-ELBOW_ANGLE # Elbow up rotated

    # Wrist Joint Angles
    wrJ_1 = theta_4_1 # Elbow down
    wrJ_2 = theta_4_2 # Elbow down
    wrJ_3 = theta_4_3 # Elbow down
    wrJ_4 = theta_4_4 # Elbow down

    # Base Joint Angles
    bJ_1 = theta_1_1 # Forward orientation
    bJ_2 = theta_1_2 # Backward orientation
    
    joint_angles = np.array([ 
                             [bJ_1, shldJ_2, elbJ_2, wrJ_2], # Elbow up
                             [bJ_1, shldJ_1, elbJ_1, wrJ_1], # Elbow down
                             [bJ_2, shldJ_4, elbJ_4, wrJ_4], # Elbow up rotated
                             [bJ_2, shldJ_3, elbJ_3, wrJ_3], # Elbow down rotated
                            ])

    return joint_angles, np.array([Xc,Yc,Zc])
    
def jointCheck (joint_angles):
    vclamp = np.vectorize(clamp)
    joint_angles = vclamp(joint_angles)

    for index,set_angle in enumerate(joint_angles):
        
        # Checks if the base angle is out of range
        if(set_angle[0] > np.pi or set_angle[0] < -np.pi):
            print("Base joint limit exceeded")
            continue
            
        # Checks if the shoulder angle is out of range
        elif(set_angle[1] > 1.9722 or set_angle[1] < -1.885):
            print("Shoulder joint limit exceeded")
            continue
            
        # Checks if the elbow angle is out of range
        elif(set_angle[2] > 1.6232 or set_angle[2] < -1.885):
            print("Elbow joint limit exceeded")
            continue

        # Checks if the wrist angle is out of range
        elif(set_angle[3] < -2.5 or set_angle[3] > 2.1468):
        #elif(set_angle[3] < -1.7453 or set_angle[3] > 2.1468):
            print("Wrist joint limit exceeded")
            continue

        # If whatever configuration passes all the checks, it will return that set_angle 
        else:
            print(set_angle)
            return set_angle,index
    
    print("None of the possible joint angles are valid.")
