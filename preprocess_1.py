# -*- coding: utf-8 -*
import numpy as np
import cv2
# from matplotlib import pyplot as plt

Min_match_count = 10
cv2.ocl.setUseOpenCL(False)
img1 = cv2.imread('test_image/original_images/canon/85.jpg')  # cv2.imread(img,0)是以灰度图的样式来读图片
img2 = cv2.imread('test_image/original_images/iphone/85.jpg')

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

# FLANN 是快速最近邻搜索包
# KTreeIndex配置索引，指定待处理核密度树的数量（理想的数量在1-16）
# search_params用它来指定递归遍历的次数。值越高结果越准 确, 但是消耗的时间也越多
# 5kd-trees和50checks总能取得合理精度，而且短时间完成

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

#-------------------------------------------------如果使用homograph
#store all the good matches as per Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance: # 最优匹配距离/次优匹配距离<0.7，即视为是匹配点
        good.append(m)
if len(good) > Min_match_count:
    # get coordinates of keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    """shape = img1.shape
    h = shape[0]
    w = shape[1]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)"""
else:
    print("Not enough matched are found %d / %d" %(len(good), Min_match_count))
    matchesMask = None

"""draw_params = dict(matchColor= (0,255,0),  # draw matches in green color
                   singlePointColor= None,
                   matchesMask= matchesMask,  # draw only inliers
                   flags= 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite("test_homo.jpg", img3)"""

#-------------------------------------------------如果不使用homograph
# Need to draw only good matches, so create a mask
"""matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]
draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=(255,0,0),
                   matchesMask=matchesMask,
                   flags=0)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
cv2.imwrite("test_no_homo.jpg", img3)"""
def getmin_max_coor(pts, matchesMask, M):
    coordinates = []
    for i in range(len(matchesMask)-1):
        if matchesMask[i] == 1:
            coordinates.append(pts[i])
    coordinates = np.float32([coordinates]).reshape(-1, 1, 2)
    a = np.max(coordinates, axis=0)[0]
    b = np.min(coordinates, axis=0)[0]
    orimax = tuple(a)
    orimin = tuple(b)
    a = np.float32(a).reshape(-1, 1, 2)
    b = np.float32(b).reshape(-1, 1, 2)
    #phomax = cv2.perspectiveTransform(a, M)
    #phomin = cv2.perspectiveTransform(b, M)
    phomax = tuple(np.float32([cv2.perspectiveTransform(a, M)]).reshape(-1, 1, 2)[0][0])
    phomin = tuple(np.float32([cv2.perspectiveTransform(b, M)]).reshape(-1, 1, 2)[0][0])
    print(orimin, orimax, phomax, phomin)
    return orimax, orimin, phomax, phomin

orimax, orimin, phomax, phomin = getmin_max_coor(src_pts, matchesMask, M)
""""# 改为分别在两张图中画框
cv2.rectangle(img1, orimin, orimax, (255, 255, 0), 3) #(255, 255, 0)是huang色,3是线的宽度
cv2.imwrite("test_1.jpg", img1)
cv2.rectangle(img2, phomin, phomax, (255, 255, 0), 3)
cv2.imwrite("test_2.jpg", img2)"""

img1 = img1[int(min(orimin[1], orimax[1])):int(max(orimin[1], orimax[1])), int(min(orimin[0], orimax[0])):int(max(orimin[0], orimax[0]))]
img2 = img2[int(min(phomin[1], phomax[1])):int(max(phomin[1], phomax[1])), int(min(phomin[0], phomax[0])):int(max(phomin[0], phomax[0]))]
img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("test_1.jpg", img1)
cv2.imwrite("test_2.jpg", img2)