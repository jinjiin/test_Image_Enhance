# -*- coding: utf-8 -*
import numpy as np
import cv2
def preprocess(img1, img2):
    Min_match_count = 10
    cv2.ocl.setUseOpenCL(False)
    img1 = cv2.imread('test_image/original_images/canon/85.jpg')  # cv2.imread(img,0)是以灰度图的样式来读图片
    img2 = cv2.imread('test_image/original_images/iphone/85.jpg')