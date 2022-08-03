# Main function
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Function to caliculate the depth map of setroscopic images

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils


if __name__ == '__main__':
    resize_factor = 0.3

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    if args.name == 'pendulum':
        img1 = cv2.imread('res/pendulum/im0.png', 0)
        img2 = cv2.imread('res/pendulum/im1.png', 0)
        K1 = np.array([[1729.05, 0, -364.24],
                       [0, 1729.05, 552.22],
                       [0, 0, 1]])
        K2 = np.array([[1729.05, 0, -364.24],
                       [0, 1729.05, 552.22],
                       [0, 0, 1]])
        baseline = 537.75
        f = 1729.05
    elif args.name == 'octagon':
        img1 = cv2.imread('res/octagon/im0.png', 0)
        img2 = cv2.imread('res/octagon/im1.png', 0)
        K1 = np.array([[1742.11, 0, 804.90],
                       [0, 1742.11, 541.22],
                       [0, 0, 1]])
        K2 = np.array([[1742.11, 0, 804.90],
                       [0, 1742.11, 541.22],
                       [0, 0, 1]])
        baseline = 221.76
        f = 1742.11
    else:
        img1 = cv2.imread('res/curule/im0.png', 0)
        img2 = cv2.imread('res/curule/im1.png', 0)
        K1 = np.array([[1758.23, 0, 977.42],
                       [0, 1758.23, 552.15],
                       [0, 0, 1]])
        K2 = np.array([[1758.23, 0, 977.42],
                       [0, 1758.23, 552.15],
                       [0, 0, 1]])
        baseline = 88.39
        f = 1758.23

    width = int(img1.shape[1] * resize_factor)
    height = int(img1.shape[0] * resize_factor)

    img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)

    while(1):
        try:
            img_keypoints, list_kp1, list_kp2 = utils.featureMapping(
                img1, img2)
            F, error = utils.ransacFMatrix([list_kp1, list_kp2])
            E = utils.getEMatrix(F, K1, K2)
            cam_pose = utils.getCameraPose(E, list_kp1)
            H1, H2, rect_pts1, rect_pts2, img1_rect, img2_rect =\
                utils.rectification(img1, img2, list_kp1, list_kp2, F)
            break
        except Exception:
            continue

    print('F Error: ', error)
    print('F Matrix is:\n', F)

    print('\nCamera Pose')
    print('Rotation:\n', cam_pose[0])
    print('Translation:\n', cam_pose[1])

    print('\nH1:\n', H1)
    print('H2:\n', H2)

    lines1 = cv2.computeCorrespondEpilines(rect_pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    left_lines, _ = utils.drawlines(img1_rect, img2_rect,
                                    lines1, rect_pts1, rect_pts2)

    lines2 = cv2.computeCorrespondEpilines(rect_pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    right_lines, _ = utils.drawlines(
        img2_rect, img1_rect, lines2, rect_pts2, rect_pts1)

    disparity_map_unscaled, disparity_map_scaled = utils.ssdCorrespondence(
        img1_rect, img2_rect)

    depth_map, depth_array = utils.disparity2Depth(
        baseline, f, disparity_map_unscaled)

    cv2.imshow('Left and Right input images', np.hstack((img1, img2)))
    cv2.imshow('img_with_keypoints', img_keypoints)
    cv2.imshow('Rectified Images With Epilines',
               np.hstack((left_lines, right_lines)))

    plt.figure(1)
    plt.title('Disparity Map Graysacle')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.figure(2)
    plt.title('Disparity Map Hot')
    plt.imshow(disparity_map_scaled, cmap='hot')
    plt.figure(3)
    plt.title('Depth Map Graysacle')
    plt.imshow(depth_map, cmap='gray')
    plt.figure(4)
    plt.title('Depth Map Hot')
    plt.imshow(depth_map, cmap='hot')

    plt.show()
    cv2.waitKey(0)
