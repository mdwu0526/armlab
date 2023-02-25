#!/bin/python
import cv2
import numpy as np
''' camera intrinsic matrix & inverse from camera_info ros message '''
K = np.array([[918.3599853515625, 0.0, 661.1923217773438],
              [0.0, 919.1538696289062, 356.59722900390625], [0.0, 0.0, 1.0]])
D = np.array([
    0.1490122675895691, -0.5096240639686584, -0.0006352968048304319, 0.0005230441456660628, 0.47986456751823425
])
''' camera intrinsics measured by checkerboard '''
# K = np.array([[971.3251488347465, 0.0, 678.0257445284223],
#               [0.0, 978.1977572681769, 374.998256815977], [0.0, 0.0, 1.0]])
# ''' Distortion Parameters from checkerboard calibration '''
# D = np.array([
#     0.13974450710810923, -0.1911712119896019, 0.004157844196335278,
#     0.0013002028638032788, 0.0
# ])

K_inv = np.linalg.inv(K)

apriltag_bundle_pos = np.array(
    [-0.013807435091499483, 0.21568357650940687, 0.9663106233532031])
apriltag_bundle_q = np.array([
    0.013333316941655229, 0.999876058474133, -0.000517708831916245,
    -0.00876391002976392
])

# points_uvd = np.array([[225, 160, 993.69], [223, 396, 995.10],
#                        [222, 680, 1000.02], [648, 160, 982.37],
#                        [648, 396, 984.71], [1077, 158, 978.16],
#                        [1079, 397, 980.94], [1081, 685, 979.85]])
points_uvd = np.array([[225, 160, 993.69], [223, 396, 995.10],
                       [222, 680, 1000.02], [648, 160, 982.37],
                       [648, 396, 984.71], [1077, 158, 978.16],
                       [1079, 397, 980.94], [1081, 685, 979.85], 
                       [430, 350, 879.68], [859, 558, 872.64], 
                       [1010, 347, 926.42], [290, 549, 942.24]])
#pixel coordinates in homogeneous coordinates
points_uv = np.delete(points_uvd, -1, axis=1)
#depths from realsense (in mm) at pixel locations
depths_camera = np.transpose(np.delete(points_uvd, (0, 1), axis=1))
#corresponding world points (mm) (not homogeneous)
# points_world = np.array([[-450, 425, 0.0], [-450, 175, 0.0], [-450, -125, 0.0],
#                          [0, 425, 0.0], [0, 175, 0.0], [450, 425, 0.0],
#                          [450, 175, 0.0], [450, -125, 0.0]])
points_world = np.array([[-450, 425, 0.0], [-450, 175, 0.0], [-450, -125, 0.0],
                         [0, 425, 0.0], [0, 175, 0.0], [450, 425, 0.0],
                         [450, 175, 0.0], [450, -125, 0.0], [-200., 225., 117.0], 
                         [200., 25, 117.0], [350., 225., 75.0 ], [-350., 25., 75.0]])
#world points in camera frame (mm) (not homogeneous)
points_ones = np.ones(depths_camera.size)

points_camera = np.transpose(
    depths_camera *
    np.dot(K_inv, np.transpose(np.column_stack((points_uv, points_ones)))))

A_ideal = np.matrix([[1., 0, 0, -14], [0, -1., 0, 195.],
                             [0, 0, -1., 985.], [0, 0, 0, 1.]])


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


def recover_homogenous_transform_pnp(world_points, image_points, camera_intrinsic_matrix, distortion_coefficients):
    '''
    Use SolvePnP to find the rigidbody transform representing the camera pose in
    world coordinates (not working)
    '''
    [_, R_exp, t] = cv2.solvePnP(world_points,
                                 image_points,
                                 camera_intrinsic_matrix,
                                 distortion_coefficients,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(R_exp)
    return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))


def recover_homogenous_transform_pnp_ransac(image_points, world_points, K):
    '''
    Use SolvePnP to find the rigidbody transform representing the camera pose in
    world coordinates (not working)
    '''
    distCoeffs = D
    [_, R_exp, t, _] = cv2.solvePnPRansac(world_points, image_points, K,
                                          distCoeffs)
    R, _ = cv2.Rodrigues(R_exp)
    return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))


def recover_homogeneous_affine_opencv(src, dst):
    _, T, _ = cv2.estimateAffine3D(src, dst, confidence=0.99)
    #print(T)
    return np.row_stack((T, (0.0, 0.0, 0.0, 1.0)))


def recover_homogenous_affine_transformation(p, p_prime):
    '''points_transformed_1 = points_transformed_1 = np.dot(
    A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))np.dot(
    A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = p[1:] - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))


def recover_homogeneous_transform_svd(m, d):
    ''' 
    finds the rigid body transform that maps m to d: 
    d == np.dot(m,R) + T
    http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    '''
    # calculate the centroid for each set of points
    d_bar = np.sum(d, axis=0) / np.shape(d)[0]
    m_bar = np.sum(m, axis=0) / np.shape(m)[0]

    # we are using row vectors, so tanspose the first one
    # H should be 3x3, if it is not, we've done this wrong
    H = np.dot(np.transpose(d - d_bar), m - m_bar)
    [U, S, V] = np.linalg.svd(H)

    R = np.matmul(V, np.transpose(U))
    # if det(R) is -1, we've made a reflection, not a rotation
    # fix it by negating the 3rd column of V
    if np.linalg.det(R) < 0:
        V = [1, 1, -1] * V
        R = np.matmul(V, np.transpose(U))
    T = d_bar - np.dot(m_bar, R)
    return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))


# # calculate A with naive extrinsic matrix
# points_transformed_ideal = np.dot(
#     np.linalg.inv(A_ideal), np.transpose(np.column_stack(
#         (points_camera, points_ones))))

#OpenCV SolvePNP calculate extrinsic matrix A
# A_pnp = recover_homogenous_transform_pnp(
#     points_uv.astype(np.float32), points_world.astype(np.float32), K, D)
# print(points_uv.astype(np.float32))
# print(points_world.astype(np.float32))
# print(A_pnp)
# points_transformed_pnp = np.dot(
#     np.linalg.inv(A_pnp), np.transpose(np.column_stack((points_camera, points_ones))))


# #OpenCV SolvePNP_Ransac calculate A
# A_pnp_r = recover_homogenous_transform_pnp_ransac(
#     points_uv.astype(np.float32), points_world.astype(np.float32), K)

# points_transformed_pnp_r = np.dot(
#     np.linalg.inv(A_pnp_r), np.transpose(np.column_stack((points_camera, points_ones))))

# # OpenCV 3Daffine calculate A (with least squares)
# # fails because we are using planar points
# A_affine_cv = recover_homogeneous_affine_opencv(
#     points_camera.astype(np.float32), points_world.astype(np.float32))
# points_transformed_affine_cv = np.dot(
#      np.linalg.inv(A_affine_cv), np.transpose(np.column_stack((points_camera, points_ones))))

# # Affine calculate A, it only takes 3 points (not least squares)
# A_affine = recover_homogenous_affine_transformation(points_world[0:8:3], points_camera[0:8:3])
# points_transformed_affine = np.dot(
#     np.linalg.inv(A_affine), np.transpose(np.column_stack((points_camera, points_ones))))

# # SVD calculate A
# A_svd = recover_homogeneous_transform_svd(points_world, points_camera)
# points_transformed_svd = np.dot(
#     np.linalg.inv(A_svd), np.transpose(np.column_stack((points_camera, points_ones))))

# # Apriltag bundle calculate A
# R_at = quaternion_rotation_matrix(apriltag_bundle_q)
# T_at = 1000.0 * apriltag_bundle_pos
# A_at = np.transpose(np.column_stack((np.row_stack(
#     (R_at, T_at)), (0, 0, 0, 1))))
# points_transformed_at = np.dot(
#     np.linalg.inv(A_at), np.transpose(np.column_stack((points_camera, points_ones))))

# """  Here we transform the camera points 
#      to world points with each extrinsic 
#      matrix and compare 
# """
# world_points = np.transpose(np.column_stack((points_world, points_ones)))

# print("\nWorld Points: \n")
# print(np.transpose(np.column_stack((points_world, points_ones))))

# print("\nREPROJECTIONS OF PIXEL LOCATIONS")
# print("\nIdeal Extrinsics: \n")
# print(A_ideal)
# print(points_transformed_ideal.astype(int))

# print("\nSolvePnP RANSAC: \n")
# print(A_pnp_r)
# print(points_transformed_pnp_r.astype(int))

# print("\nSolvePnP: \n")
# print(A_pnp)
# print(points_transformed_pnp.astype(int))

# print("\nOpenCV 3D Affine: \n")
# print(A_affine_cv)
# print(points_transformed_affine_cv.astype(int))

# print("\n3D Affine (StackExchange): \n")
# print(A_affine)
# print(points_transformed_affine.astype(int))

# print("\nRigid Body (SVD): \n")
# print(A_svd)
# print(points_transformed_svd.astype(int))

# print("\nApriltag Bundle: \n")
# print(A_at)
# print(points_transformed_at.astype(int))