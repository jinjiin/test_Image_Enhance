# -*- coding: utf-8 -*
import numpy as np
import cv2
import os

def preprocess(dir1, dir2, dir3, dir4, img1num):
    Min_match_count = 10
    cv2.ocl.setUseOpenCL(False)
    img1 = cv2.imread(dir1 + '/' + img1num + '.jpg')
    img2 = cv2.imread(dir2 + '/' + img1num + '.jpg')
    # img1 = cv2.imread('test_image/original_images/canon/' + img1num+'.jpg')  # cv2.imread(img,0)是以灰度图的样式来读图片
    # img2 = cv2.imread('test_image/original_images/iphone/' + img1num+'.jpg')
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
        phomax = tuple(np.float32([cv2.perspectiveTransform(a, M)]).reshape(-1, 1, 2)[0][0])
        phomin = tuple(np.float32([cv2.perspectiveTransform(b, M)]).reshape(-1, 1, 2)[0][0])
        print(orimin, orimax, phomax, phomin)
        return orimax, orimin, phomax, phomin

    orimax, orimin, phomax, phomin = getmin_max_coor(src_pts, matchesMask, M)
    img1 = img1[int(min(orimin[1], orimax[1])):int(max(orimin[1], orimax[1])),
           int(min(orimin[0], orimax[0])):int(max(orimin[0], orimax[0]))]
    img2 = img2[int(min(phomin[1], phomax[1])):int(max(phomin[1], phomax[1])),
           int(min(phomin[0], phomax[0])):int(max(phomin[0], phomax[0]))]
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dir3 + '/' + img1num+'.jpg', img1)
    cv2.imwrite(dir4 + '/' + img1num+'.jpg', img2)


def getfilenames(dir):
    files = os.listdir(dir)
    filenames = []
    for i in files:
        filenames.append(i.split('.')[0])
    fragment = len(filenames)/5
    return filenames, fragment
if __name__=='__main__':
    dir1 = ''
    dir2 = ''
    dir3 = ''
    dir4 = ''
    filenames, fragment = getfilenames('test_image\\original_images\\canon')
    for i in filenames[: fragment]:
        preprocess(dir1, dir2, dir3, dir4, i)

    """p = mp.pool(processes=10)
    p.map_async(preprocess, filenames)
    p.close()
    p.join()"""