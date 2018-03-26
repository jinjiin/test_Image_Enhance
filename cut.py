# -*- coding: utf-8 -*
import cv2
import numpy as np
import math
import os
from multiprocessing import Pool

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
            print(i, j)
            frag = img1[weidth1: weidth2, height1: height2]
            frag1 = img2[weidth1: weidth2, height1: height2]
            args = []
            movelen = 10
            args.append(frag1)
            if height1+movelen <= height and height2+movelen <= height:  # 其实后面的height1+movelen<=height可以去掉
                frag2 = img2[weidth1: weidth2, height1+movelen: height2+movelen]  # down
                args.append(frag2)
            if weidth2+movelen < weidth and height1-movelen >= 0:
                frag3 = img2[weidth1+movelen: weidth2+movelen, height1-movelen: height2-movelen]  # right up
                args.append(frag3)
            if weidth2+movelen < weidth:
                frag4 = img2[weidth1+movelen: weidth2+movelen, height1: height2]  # right
                args.append(frag4)
            if weidth2+movelen < weidth and height2+movelen < height:  # right down
                frag5 = img2[weidth1+movelen: weidth2+movelen, height1+movelen: height2+movelen]
                args.append(frag5)
            args1 = []
            for l in range(len(args)):
                args1.append(frag)

            #p = ProcessingPool(5)
            p = Pool(5)
            results = p.map(NCC_colors, args1, args)

            maxflag = -1
            max = -2
            for k in range(len(results)-1):
                if results[k] > max:
                    max = results[k]
                    maxflag = k
            if results[max] > 0.55:
                cv2.imwrite('patches/sony/sony/' + str(picnum) + ".jpg", args[maxflag])
                cv2.imwrite('patches/sony/canon/' + str(picnum) + ".jpg", frag)
                picnum = picnum + 1
                print(picnum)
            """for i in range(len(args)):
                print(NCC(args1[i], args[i]))"""
    p.close()
    p.join()
    return picnum

def NCC_grey(img1, img2):
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
def NCC_colors(img1, img2):
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    mean1 = np.mean(img1, axis=(0, 1))
    mean2 = np.mean(img2, axis=(0, 1))
    """weidth = img1.shape[0]
    height = img1.shape[1]
    for i in range(weidth - 1):
        for j in range(height - 1):
            img1[i][j] = img1[i][j] - mean1
            img2[i][j] = img2[i][j] - mean2"""
    img1 = img1 - mean1
    img2 = img2 - mean2
    A = np.sum(img1 * img2)
    B = np.sum(img1 * img1)
    C = np.sum(img2 * img2)
    return A/(math.sqrt(B)*math.sqrt(C))

def getfilenames(dir):
    files = os.listdir(dir)
    filenames = []
    for i in files:
        filenames.append(i.split('.')[0])
    return filenames
if __name__ == '__main__':
    filenames = getfilenames('resize/sony/sony')
    patchnum = 0
    for num in filenames:
        img1 = cv2.imread('resize/sony/sony/' + str(num) + '.jpg')
        img2 = cv2.imread('resize/sony/canon/' + str(num) + '.jpg')
        patchnum = detect_pathes(img1, img2, patchnum)