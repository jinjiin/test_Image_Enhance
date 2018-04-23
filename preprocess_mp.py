# -*- coding: utf-8 -*
import re

import numpy as np
import cv2
import os
import multiprocess as mp
import scipy.misc

def preprocess(img1num):
    Min_match_count = 10
    cv2.ocl.setUseOpenCL(False)
    if os.path.exists('dped/mi/resize/test/mi/' + str(img1num) + '.jpg'):
        print(str(img1num) + '.jpg have existed!')
        return 0
    img1 = cv2.imread('dped/mi/full_test_data/full_all/mi/' + str(img1num) + '.jpg')  # cv2.imread(img,0)是以灰度图的样式来读图片
    img2 = cv2.imread('dped/mi/full_test_data/full_all/canon/' + str(img1num) + '.jpg')
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    Flann_index_kdtree = 0
    index_params = dict(algorithm=Flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 最优匹配距离/次优匹配距离<0.7，即视为是匹配点
            good.append(m)
    if len(good) > Min_match_count:
        # get coordinates of keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matched are found %d / %d" % (len(good), Min_match_count))
        matchesMask = None

    def getmin_max_coor(pts, matchesMask, M):
        coordinates = []
        for i in range(len(matchesMask) - 1):
            if matchesMask[i] == 1:
                coordinates.append(pts[i])
        coordinates = np.float32([coordinates]).reshape(-1, 1, 2)
        a = np.max(coordinates, axis=0)[0]
        b = np.min(coordinates, axis=0)[0]
        orimax = tuple(a)
        orimin = tuple(b)
        a = np.float32(a).reshape(-1, 1, 2)
        b = np.float32(b).reshape(-1, 1, 2)
        # phomax = cv2.perspectiveTransform(a, M)
        # phomin = cv2.perspectiveTransform(b, M)
        phomax = tuple(np.float32([cv2.perspectiveTransform(a, M)]).reshape(-1, 1, 2)[0][0])
        phomin = tuple(np.float32([cv2.perspectiveTransform(b, M)]).reshape(-1, 1, 2)[0][0])
        print(orimin, orimax, phomax, phomin)
        return orimax, orimin, phomax, phomin

    orimax, orimin, phomax, phomin = getmin_max_coor(src_pts, matchesMask, M)
    img1 = img1[int(min(orimin[1], orimax[1])):int(max(orimin[1], orimax[1])),
           int(min(orimin[0], orimax[0])):int(max(orimin[0], orimax[0]))]
    img2 = img2[int(min(phomin[1], phomax[1])):int(max(phomin[1], phomax[1])),
           int(min(phomin[0], phomax[0])):int(max(phomin[0], phomax[0]))]
    #img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_CUBIC)
    img1 = scipy.misc.imresize(img1, (img2.shape[0], img2.shape[1]), interp='cubic')
    cv2.imwrite('dped/mi/resize/test/mi/' + str(img1num) + '.jpg', img1)
    cv2.imwrite('dped/mi/resize/test/canon/' + str(img1num) + '.jpg', img2)

def getfilenames(dir):
    files = os.listdir(dir)
    filenames = []
    for i in files:
        filenames.append(i.split('.')[0])
    return filenames
# rename('D:\采集数据\test\Camera')
def rename(dir):
    files = os.listdir(dir)
    for i in files:
        print(re.findall(r'[^()]+', i)[1])
        os.rename(dir + '\\' + i, dir + '\\' + re.findall(r'[^()]+', i)[1] + '.jpg')
if __name__=='__main__':
    """filenames = getfilenames('dped/mi/full_training_data/mi')
    print(filenames)"""
    filename = [12, 17, 18, 22, 23, 24, 37, 38, 39, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 83, 84, 85, 86, 87, 97, 98, 99, 100]
    p = mp.Pool(processes=10)
    p.map_async(preprocess, filename)
    p.close()
    p.join()

