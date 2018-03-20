import cv2
import numpy as np

img2 = cv2.imread('test_image\\original_images\\canon\\85.jpg')
shape = img2.shape
weidth = shape[0]
height = shape[1]
print(weidth, height)
if weidth % 100 == 0:
    maxi = int(weidth/100)
else:
    maxi = int(weidth/100 + 1)
if height % 100 == 0:
    maxj = int(height/100)
else:
    maxj = int(height/100 + 1)
num = 0
for i in range(1, maxi):
    for j in range(1, maxj):
        if i < maxi:
            weidth1 = (i-1)*100
            weidth2 = i*100
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
        cv2.imwrite(str(num)+".jpg", img)
        num = num + 1