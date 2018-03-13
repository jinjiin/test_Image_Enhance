# -*- coding: utf-8 -*
import numpy as np
import cv2
# from matplotlib import pyplot as plt

Min_match_count = 10
cv2.ocl.setUseOpenCL(False)
img1 = cv2.imread('test_image/original_images/canon/85.jpg')  # cv2.imread(img,0)是以灰度图的样式来读图片
img2 = cv2.imread('test_image/original_images/iphone/85.jpg')
print(img1)
print(img1.shape)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SIFT() #opencv version > 2.4.9

# find the keypoints and descriptors with SIFT
# Each keypoint is a special structure which has many attributes like its (x,y) coordinates,
# size of the meaningful neighbourhood, angle which specifies its orientation,
# response that specifies strength of keypoints etc.
# kp will be a list of keypoints and des is a numpy array of shape Number_of_Keypoints×128.

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
Flann_index_kdtree = 0
index_params = dict(algorithm=Flann_index_kdtree, trees=5)
search_params = dict(checks=50)

src_pts = np.float32([kp1[i].pt for i in range(len(kp1))]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[i].pt for i in range(len(kp2))]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

h, w = img1.shape
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
cv2.imwrite("test_1.jpg", img2)

# get coordinates of keypoints
"""src_pts = np.float32([kp1[m.queryIdx].pt for m,n in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m,n in matches]).reshape(-1, 1, 2)
print("src_pts")
print(src_pts)
print("dst_pts")
print(dst_pts)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]
draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=(255,0,0),
                   matchesMask=matchesMask,
                   flags=0)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
print(np.min(src_pts, axis=1))
print(np.min(src_pts, axis=1)[0][0])  # 1508.54
print(np.min(src_pts, axis=1)[0][1])  # 2.6525
print(np.max(src_pts, axis=1))
print(np.max(src_pts, axis=1)[0][0])  # 1508.54
print(np.max(src_pts, axis=1)[0][1])  # 2.6525
# 改为分别在两张图中画框
cv2.rectangle(img1, (np.min(src_pts, axis=1)[0][0], np.min(src_pts, axis=1)[0][1]),
               (np.max(src_pts, axis=1)[0][0], np.max(src_pts, axis=1)[0][1]),
               (255, 255, 0), 3) #(255, 255, 0)是huang色,3是线的宽度
cv2.rectangle(img2, (np.min(dst_pts, axis=1)[0][0], np.min(dst_pts, axis=1)[0][1]),
               (np.max(dst_pts, axis=1)[0][0], np.max(dst_pts, axis=1)[0][1]),
               (255, 255, 0), 3) #(255, 255, 0)是huang色,3是线的宽度
cv2.imwrite("test_1.jpg", img1)
cv2.imwrite("test_2.jpg", img2)"""