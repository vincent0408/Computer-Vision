import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping
import matplotlib.pyplot as plt

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        # TODO: 1.feature detection & matching
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        max_dist_cut = 100
        matches = sorted(matches, key = lambda x:x.distance)
        query_data = np.vstack((np.array([kp1[x.queryIdx].pt for x in matches]).T, np.ones(len(matches))[None, :]))
        train_data = np.vstack((np.array([kp2[x.trainIdx].pt for x in matches]).T, np.ones(len(matches))[None, :]))
        N = 1000
        inliers = 0
        # TODO: 2. apply RANSAC to choose best H
        for i in range(N):
            random_idx = random.sample(range(len(matches)), 13)
            query_corners = query_data[:,random_idx][:2].T
            train_corners = train_data[:,random_idx][:2].T
            H = solve_homography(train_corners, query_corners)
            predicted_query_data = np.matmul(H, train_data)
            predicted_query_data = predicted_query_data / predicted_query_data[2]
            error = (predicted_query_data - query_data)[:2]
            error_norm = np.linalg.norm(error, 1, axis=0)
            predicted_inliers = len(np.nonzero(error_norm < 4))
            if(predicted_inliers > inliers):
                inliers = predicted_inliers
                best_H = H


        # TODO: 3. chain the homographies
        last_best_H = np.matmul(best_H, last_best_H)
        # TODO: 4. apply warping
        dst = warping(im1, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return dst

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)