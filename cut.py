# -*- coding: utf-8 -*
import cv2
import numpy as np
from numpy.fft import irfft, rfft
import math


def cut():
    img2 = cv2.imread('cut_images\\cut_1.jpg')
    shape = img2.shape
    weidth = shape[0]
    height = shape[1]
    print(weidth, height)
    if weidth % 100 == 0:
        maxi = int(weidth / 100)
    else:
        maxi = int(weidth / 100 + 1)
    if height % 100 == 0:
        maxj = int(height / 100)
    else:
        maxj = int(height / 100 + 1)
    num = 0
    for i in range(1, maxi):
        for j in range(1, maxj):
            if i < maxi:
                weidth1 = (i - 1) * 100
                weidth2 = i * 100
            elif i == maxi:
                weidth1 = weidth - 100
                weidth2 = weidth
            if j < maxj:
                height1 = (j - 1) * 100
                height2 = j * 100
            elif j == maxj:
                height1 = height - 100
                height2 = height
            img = img2[weidth1:weidth2, height1:height2]
            cv2.imwrite('cut_images\\canon\\' + str(num) + ".jpg", img)
            num = num + 1


def rotation():
    img = cv2.imread('cut_images\\canon\\63.jpg', 0)
    img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
    h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全0图像
    bins = np.arange(256).reshape(256, 1)  # 直方图中各bin的顶点位置
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色
    for ch, col in enumerate(color):
        originHist = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)
        hist = np.int32(np.around(originHist))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
        h = np.flipud(h)
        cv2.imshow('colorhist', h)
        cv2.waitKey(0)


def rotation(img):
    result = cv2.calcHist([img],
                          [0],  # 使用的通道
                          None,  # 没有使用mask
                          [256],  # HistSize
                          [0.0, 255.0])  # 直方图柱的范围
    print(result)
    result2 = cv2.calcHist([img2],
                           [0],  # 使用的通道
                           None,  # 没有使用mask
                           [256],  # HistSize
                           [0.0, 255.0])  # 直方图柱的范围
    print('---------------------')
    print(result2)

def NCC(frag1, frag2):
    img1 = cv2.imread('cut_images\\canon\\63.jpg', 0)
    img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
    print(img1)
    mean1 = np.mean(img1, axis=(0, 1))
    print(mean1)
    mean2 = np.mean(img2, axis=(0, 1))
    weidth = img1.shape[0]
    height = img1.shape[1]
    A = 0
    B = 0
    C = 0
    for i in range(0, weidth-1):
        for j in range(0, height-1):
            img1[i][j] = img1[i][j] - mean1
            img2[i][j] = img2[i][j] - mean2
            A = img1[i][j] * img1[i][j] + A
            B = img2[i][j] * img2[i][j] + B
            C = img1[i][j] * img2[i][j] + C

    return C/(math.sqrt(A)*math.sqrt(B))
    """A = np.sum(img1 * img2)
    B = np.sum(img1 * img1) * np.sum(img2 * img2)
    return A/(np.sqrt(B))"""
def test_NCC(x, y):
    xcorr = lambda x, y: irfft(rfft(x) * rfft(y[::-1]))
    img1 = cv2.imread('cut_images\\canon\\63.jpg', 0)
    img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
    print(xcorr(img1, img2))
    return xcorr
if __name__ == '__main__':
    print(NCC(1, 2))