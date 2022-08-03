# Utils Library
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Function definetions

import cv2
import random
import numpy as np


def featureMapping(gray1, gray2, no_of_keyPoints=40):
    """
    Definition
    ---
    Function to map the the similar festures in both the images

    Parameters
    ---
    gray1, gray2: grayscale images
    no_of_keyPoints: number of matching features (default = 40)

    Returns
    ---
    img_keypoints: input image 1 and 2 with keypoints marked
    list_kp1, list_kp2: List of key points in image 1 and 2 respectively
    """
    orb = cv2.ORB_create(nfeatures=10000)
    # find the keypoints and descriptors with ORB
    kp1, d1 = orb.detectAndCompute(gray1, None)
    kp2, d2 = orb.detectAndCompute(gray2, None)
    # Creating a brute force matcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.match(d1, d2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Select first 30 matches.
    final_matches = matches[:no_of_keyPoints]

    img_keypoints = cv2.drawMatches(
        gray1, kp1, gray2, kp2, final_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Getting x,y coordinates of the matches
    list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in final_matches]
    list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in final_matches]

    return img_keypoints, list_kp1, list_kp2


def getFMatrix(list1, list2):
    """
    Definition
    ---
    Function to generate F matrix based on given key points

    Parameters
    ---
    list1, list2: list of points

    Returns
    ---
    F: F Matrix
    """
    A = np.zeros(shape=(len(list1), 9))
    for i in range(len(list1)):
        x1, y1 = list1[i][0], list1[i][1]
        x2, y2 = list2[i][0], list2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    U, s, Vt = np.linalg.svd(A)
    F = Vt[-1, :]
    F = F.reshape(3, 3)

    Uf, Df, Vft = np.linalg.svd(F)
    Df[2] = 0
    s = np.zeros((3, 3))
    for i in range(3):
        s[i][i] = Df[i]
    F = np.dot(Uf, np.dot(s, Vft))
    return F


def ransacFMatrix(lists_of_kp, max_inliers=20, threshold=0.05):
    """
    Definition
    ---
    Find the best fitting line between the two given list of points using
    RANSAC

    Parameters
    ---
    lists_of_kp: lists of front_points
    max_inliers: max number of inliers (default = 20)
    threshold: threshold for F matrix distance

    Returns
    ---
    Best_F: Best F Matrix for given set of key points
    F_Error: Error of the generated F matrix
    """
    list_kp1 = lists_of_kp[0]
    list_kp2 = lists_of_kp[1]
    pairs = list(zip(list_kp1, list_kp2))
    for i in range(1000):
        pairs = random.sample(pairs, 8)
        rd_list_kp1, rd_list_kp2 = zip(*pairs)
        F = getFMatrix(rd_list_kp1, rd_list_kp2)
        tmp_inliers_img1 = []
        tmp_inliers_img2 = []
        for i in range(len(list_kp1)):
            img1_x = np.array([list_kp1[i][0], list_kp1[i][1], 1])
            img2_x = np.array([list_kp2[i][0], list_kp2[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F, img1_x)))

            if distance < threshold:
                tmp_inliers_img1.append(list_kp1[i])
                tmp_inliers_img2.append(list_kp2[i])

        num_of_inliers = len(tmp_inliers_img1)
        if num_of_inliers > max_inliers:
            max_inliers = num_of_inliers
            Best_F = F
            F_Error = distance
    return Best_F, F_Error


def getEMatrix(F, K1, K2):
    """
    Definition
    ---
    Function to genreate E Matrix

    Parameters
    ---
    F: F Matrix
    K1: camera 1 internsic matrix
    K2: camera 2 internsic matrix

    Returns
    ---
    Best_F: Best F Matrix for given set of key points
    F_Error: Error of the generated F matrix
    """
    E = np.dot(K2.T, np.dot(F, K1))
    return E


def getCameraPose(E, list_kp1):
    """
    Definition
    ---
    Function to get the best camera pose for the two images

    Parameters
    ---
    E: E matrix for the two Images
    list_kp1: list of key points of 1st image

    Returns
    ---
    camera_pose: Best camera pose for the two images
    """
    U, s, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    C1, C2 = U[:, 2], -U[:, 2]
    R1, R2 = np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(W.T, Vt))
    camera_poses = [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]

    max_len = 0
    for pose in camera_poses:
        front_points = []
        for point in list_kp1:
            X = np.array([point[0], point[1], 1])
            V = X - pose[1]

            condition = np.dot(pose[0][2], V)
            if condition > 0:
                front_points.append(point)

        if len(front_points) > max_len:
            max_len = len(front_points)
            camera_pose = pose
    return camera_pose


def rectification(gray1, gray2, list_kp1, list_kp2, F):
    """
    Definition
    ---
    Function to perform rectification on the two images

    Parameters
    ---
    gray1, gray2: grayscale images
    list_kp1, list_kp2: List of key points in image 1 and 2 respectively
    F: F Matrix

    Returns
    ---
    H1: Homography matrix of image 1
    H2: Homography matrix of image 2
    rect_pts1: Rectified key points of image 1
    rect_pts2: Rectified key points of image 2
    gray1_rect: rectified image 1
    gray2_rect: rectified image 2
    """
    pts1 = np.int32(list_kp1)
    pts2 = np.int32(list_kp2)

    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    rect_pts1 = np.zeros((pts1.shape), dtype=int)
    rect_pts2 = np.zeros((pts2.shape), dtype=int)
    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rect_pts1[i] = new_point1

        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rect_pts2[i] = new_point2

    gray1_rect = cv2.warpPerspective(gray1, H1, (w1, h1))
    gray2_rect = cv2.warpPerspective(gray2, H2, (w2, h2))

    return H1, H2, rect_pts1, rect_pts2, gray1_rect, gray2_rect


def drawlines(gray1, gray2, lines, pts1src, pts2src):
    """
    Definition
    ---
    Function to draw the epipolar lines

    Parameters
    ---
    gray1, gray2: grayscale images
    lines: Epipolar lines
    pts1src: source of rectified points from image 1
    pts2src: source of rectified points from image 2

    Returns
    ---
    img1_lines: image 1 with epipolar lines
    img2_lines: image 2 with epipolar lines
    """
    r, c = gray1.shape
    img1_lines = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    img2_lines = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_lines = cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        img1_lines = cv2.circle(img1_lines, tuple(pt1), 5, color, -1)
        img2_lines = cv2.circle(img2_lines, tuple(pt2), 5, color, -1)
    return img1_lines, img2_lines


def getSSD(val1, val2):
    """
    Definition
    ---
    Function to calucalte sum of squared distances between 2 points

    Parameters
    ---
    val1, val2: points of interest

    Returns
    ---
    ssd: computed sum of square distances between 2 points
    """
    if val1.shape != val2.shape:
        return -1
    return np.sum((val1 - val2)**2)


def blockCompare(y, x, l_block, r_array, b_size, x_search_size, y_search_size):
    """
    Definition
    ---
    Block comparison function used for comparing windows on left and right
    images and find the minimum value ssd match the pixels

    Returns
    ---
    min_index: minimum value ssd
    """
    x_min = max(0, x - x_search_size)
    x_max = min(r_array.shape[1], x + x_search_size)
    y_min = max(0, y - y_search_size)
    y_max = min(r_array.shape[0], y + y_search_size)

    first = True
    min_ssd = None
    min_index = None
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            r_block = r_array[y: y+b_size, x: x+b_size]
            ssd = getSSD(l_block, r_block)
            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)
    return min_index


def ssdCorrespondence(gray1, gray2):
    """
    Definition
    ---
    Correspondence applied on the whole image to compute the disparity map and
    finally disparity map is scaled

    Parameters
    ---
    gray1, gray2: grayscale images

    Returns
    ---
    disparity_map_unscaled: Unscaled disparity map
    disparity_map_scaled: scaled disparity map
    """
    b_size = 15
    x_search_size = 50
    y_search_size = 1
    h, w = gray1.shape
    disparity_map = np.zeros((h, w))

    for y in range(b_size, h-b_size):
        for x in range(b_size, w-b_size):
            l_block = gray1[y:y + b_size, x:x + b_size]
            index = blockCompare(y, x, l_block, gray2,
                                 b_size, x_search_size, y_search_size)
            disparity_map[y, x] = abs(index[1] - x)

    disparity_map_unscaled = disparity_map.copy()
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_map[i][j] = int(
                (disparity_map[i][j]*255)/(max_pixel-min_pixel))

    disparity_map_scaled = disparity_map
    return disparity_map_unscaled, disparity_map_scaled


def disparity2Depth(baseline, f, img):
    """
    Definition
    ---
    This is used to compute the depth values from the disparity map

    Parameters
    ---
    baseline: baseline of camera
    f: focal length of camera
    img: unscaled disparity map

    Returns
    ---
    depth_map: depth max
    depth_array: depth array
    """
    depth_map = np.zeros((img.shape[0], img.shape[1]))
    depth_array = np.zeros((img.shape[0], img.shape[1]))

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/img[i][j]
            depth_array[i][j] = baseline*f/img[i][j]
    return depth_map, depth_array
