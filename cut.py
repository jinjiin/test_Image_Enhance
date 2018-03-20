import cv2
import numpy as np

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


img = cv2.imread('cut_images\\canon\\63.jpg', 0)
img2 = cv2.imread('cut_images\\iphone\\63.jpg', 0)
"""h = np.zeros((256,256,3)) #创建用于绘制直方图的全0图像
bins = np.arange(256).reshape(256,1) #直方图中各bin的顶点位置
color = [(255,0,0),(0,255,0),(0,0,255)] #BGR三种颜色
for ch, col in enumerate(color):
    originHist = cv2.calcHist([img],[ch],None,[256],[0,256])
    cv2.normalize(originHist, originHist,0,255*0.9,cv2.NORM_MINMAX)
    hist=np.int32(np.around(originHist))
    pts = np.column_stack((bins,hist))
    cv2.polylines(h,[pts],False,col)
    h=np.flipud(h)
    cv2.imshow('colorhist',h)
    cv2.waitKey(0)"""
"""result = cv2.calcHist([img],
    [0], #使用的通道
    None, #没有使用mask
    [256], #HistSize
    [0.0,255.0]) #直方图柱的范围
print(result)
result2 = cv2.calcHist([img2],
    [0], #使用的通道
    None, #没有使用mask
    [256], #HistSize
    [0.0,255.0]) #直方图柱的范围
print('---------------------')
print(result2)"""
