from nis import match
import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    print("this is points2d.shape:")
    print(points2d.shape) # (24, 2)
    print("this is points3d.shape:")
    print(points3d.shape) # (24, 3)
    ########################
    # TODO: Your code here #
    ########################
    # This M matrix came from a call to rand(3,4). It leads to a high residual.

    def create_A(points2d, points3d):
        (X, Y, Z) = points3d
        (u, v) = points2d 
        A = np.array([[X,Y,Z,1,0,0,0,0,-X*u,-Y*u,-Z*u],
                      [0,0,0,0,X,Y,Z,1,-X*v,-Y*v,-Z*v]])
        return A 

    A = []
    for i in range(points2d.shape[0]):
        one_A = create_A(points2d[i],points3d[i])
        A.append(one_A)
    A = np.array(A)
    # print("this is A.shape:")
    # print(A.shape) # (24, 2, 11)

    A = np.reshape(A, (-1, 11))
    # print("this is A after reshape:")
    # print(A.shape) # (48, 11)

    flatten_2d = np.reshape(points2d, (-1,1))
    # print("this is points2d after reshape:")
    # print(flatten_2d.shape) # (48, 1)

    M = np.linalg.lstsq(A, flatten_2d)[0]
    M = np.concatenate((M, np.array([[1]])))
    M = np.reshape(M, (3,4))
    print(M)

    return M 

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    ########################

    # This is an intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])
    all_A = []
    for i in range(points1.shape[0]):
        u = points1[i][0]
        v = points1[i][1]
        u2 = points2[i][0]
        v2 = points2[i][1]
        A = np.array([u*u2, v*u2, u2, u*v2, v*v2, v2, u, v, 1])
        all_A.append(A)
    U,S,Vh = np.linalg.svd(all_A)
    F = Vh[-1,:]
    F = np.reshape(F, (3,3))
    U2, S2, Vh2 = np.linalg.svd(F)
    S2[-1] = 0
    F = U2 @ np.diagflat(S2) @ Vh2


    return F

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would call the function that estimates the 
    fundamental matrix (either the "cheat" function or your own 
    estimate_fundamental_matrix) iteratively within this function.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.
    best_inliner = 0
    best_indices = []
    threshold = 0.01
    best_f = np.ones((3,3))

   # matches1_homo = np.concatenate((matches1, np.ones((matches1.shape[0],1))), axis = -1)
   # matches2_homo = np.concatenate((matches2, np.ones((matches2.shape[0],1))), axis = -1)  

    for i in range(num_iters):
        inliner = 0
        inliner_list = []
        random_index = np.random.randint(matches1.shape[0], size = 9)
        best_F = estimate_fundamental_matrix(matches1[random_index,:], matches2[random_index,:])
        # best_F, _ = cv2.findFundamentalMat(matches1[random_index,:], matches2[random_index,:], cv2.FM_8POINT, 1e10, 0, 1)

        for l in range(matches1.shape[0]): # test whether each point is within the threshold 
            distance = np.abs(np.matmul(np.matmul(np.transpose(np.append(matches1[l,:],1)),best_F), np.append(matches2[l,:],1)))
            # convert matches1 and matches2 to homogeneous coordinates 
            # definition for the fundamental matrix: x_tranpose F x' = 0
            if distance < threshold:
                inliner_list.append(l)
                inliner = inliner + 1
        if inliner > best_inliner:
            best_inliner = inliner
            best_f = best_F
            best_indices = inliner_list

        final_inliners1 = matches1[best_indices,:]
        final_inliners2 = matches2[best_indices,:]
    
    return best_f, final_inliners1, final_inliners2


def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    ########################
    # TODO: Your code here #

    # Fill in the correct shape
    points3d = []

    # Solve for ground truth points
    for i in range(points1.shape[0]):
        haha = np.concatenate((M1[0:2,0:3], M2[0:2, 0:3]), axis=0)
        # print(m1.shape) # (4, 3)
        first = np.array([[M1[2,0]*points1[i][0], M1[2,0]*points1[i][1], M2[2,0]*points2[i][0], M2[2,0]*points2[i][1]]])
        # print(first.shape) # (1, 4)
        second = np.array([[M1[2,1]*points1[i][0], M1[2,1]*points1[i][1], M2[2,1]*points2[i][0], M2[2,1]*points2[i][1]]])
        third = np.array([[M1[2,2]*points1[i][0], M1[2,2]*points1[i][1], M2[2,2]*points2[i][0], M2[2,2]*points2[i][1]]])
        final = np.concatenate((np.transpose(first), np.transpose(second), np.transpose(third)), axis = 1)
        # print(final.shape) # (4, 3)
        A = haha - final
        fourth = np.array([[M1[2,3]*points1[i][0], M1[2,3]*points1[i][1], M2[2,3]*points2[i][0], M2[2,3]*points2[i][1]]])
        yaya = [M1[0,3],M1[1,3],M2[0,3],M2[1,3]]
        b = fourth - yaya
        b = np.transpose(b)
        # print(b.shape) # (4, 1)
        answers = np.linalg.lstsq(A,b)[0]
        answers = np.reshape(answers, (1,-1))
        points3d.append(answers)

    points3d = np.array(points3d)
    points3d = np.squeeze(points3d) # (135, 3)
    print(points3d.shape)
        
    ########################

    return points3d
