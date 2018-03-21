# -*- coding: utf-8 -*
import cv2
import numpy as np
from numpy.fft import irfft, rfft
import math
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool

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

def detect_pathes(img1, img2, picnum):
    global frag2, frag3, frag4, frag5
    shape = img1.shape
    weidth = shape[0]
    height = shape[1]
    maxi = int(weidth/100)
    maxj = int(height/100)
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
            frag = img1[weidth1: weidth2, height1: height2]
            frag1 = img2[weidth1: weidth2, height1: height2]
            args = []
            args.append((frag, frag1))
            if height1-50 >= 0 and height2-50 >= 0:  # 其实后面的height2-50>=0可以去掉
                frag2 = img2[weidth1: weidth2, height1-50: height2-50]  # down
                args.append((frag, frag2))
            if weidth2+50 < weidth and height1-50 >= 0:
                frag3 = img2[weidth1+50: weidth2+50, height1-50: height2-50]  # right up
                args.append((frag, frag3))
            if weidth2+50 < weidth:
                frag4 = img2[weidth1+50: weidth2+50, height1: height2]  # right
                args.append((frag, frag4))
            if weidth2+50 < weidth and height2+50 < height:  # right down
                frag5 = img2[weidth1+50: weidth2+50, height1+50: height2+50]
                args.append((frag, frag5))
            args = [(frag, frag1), (frag, frag2), (frag, frag3), (frag, frag4), (frag, frag5)]
            p = Pool(5)
            results = p.map(for_mp_pack(args))
            p.close()
            p.join()
            for i in range(len(results)-1):
                if results[i] > 0.9:
                    if i == 0:
                        cv2.imwrite('cut_image_2\\iphone\\' + str(picnum) + ".jpg", args[i][1])
                        cv2.imwrite('cut_image_2\\canon\\' + str(picnum) + ".jpg", frag)
                    elif i == 1:
                        cv2.imwrite('cut_image_2\\iphone\\' + str(picnum) + ".jpg", frag2)
                        cv2.imwrite('cut_image_2\\canon\\' + str(picnum) + ".jpg", frag)
                    elif i == 2:
                        cv2.imwrite('cut_image_2\\iphone\\' + str(picnum) + ".jpg", frag3)
                        cv2.imwrite('cut_image_2\\canon\\' + str(picnum) + ".jpg", frag)
                    elif i == 3:
                        cv2.imwrite('cut_image_2\\iphone\\' + str(picnum) + ".jpg", frag4)
                        cv2.imwrite('cut_image_2\\canon\\' + str(picnum) + ".jpg", frag)
                    elif i == 4:
                        cv2.imwrite('cut_image_2\\iphone\\' + str(picnum) + ".jpg", frag5)
                        cv2.imwrite('cut_image_2\\canon\\' + str(picnum) + ".jpg", frag)

def for_mp_pack(args):
    NCC(args[0], args[1])

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

def NCC(img1, img2):
    # img1 = cv2.imread('cut_images\\canon\\63.jpg', 0)
    # img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
    # mean1 = np.mean(img1, axis=(0, 1))
    # mean2 = np.mean(img2, axis=(0, 1))
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    mean1 = img1.mean()
    mean2 = img2.mean()
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
def test_NCC(x, y):
    xcorr = lambda x, y: irfft(rfft(x) * rfft(y[::-1]))
    img1 = cv2.imread('cut_images\\canon\\63.jpg', 0)
    img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
    print(xcorr(img1, img2))
    return xcorr

def conv2(a,b):
    ma,na = a.shape
    mb,nb = b.shape
    return np.fft.ifft2(np.fft.fft2(a,[2*ma-1,2*na-1])*np.fft.fft2(b,[2*mb-1,2*nb-1]))

# compute a normalized 2D cross correlation using convolutions
# this will give the same output as matlab, albeit in row-major order
def normxcorr2(b,a):
    c = conv2(a, np.flipud(np.fliplr(b)))
    a = conv2(a ** 2, np.ones(b.shape))
    b = sum(b.flatten()**2)
    c = c / np.sqrt(a * b)
    return c
if __name__ == '__main__':
    """img1 = cv2.imread('cut_images\\canon\\63.jpg', 0)
    img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
    img1 = np.reshape(img1, [1, 100*100])
    img2 = np.reshape(img2, [1, 100*100])
    print(normxcorr2(img1, img2))"""
    img1 = cv2.imread('cut_images\\cut_1.jpg', 0)
    img2 = cv2.imread('cut_images\\cut_2.jpg', 0)
    detect_pathes(img1, img2, 0)