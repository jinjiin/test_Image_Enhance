import cv2
def compare(img1, img2):
    img1 = cv2.imread(img1)  # cv2.imread(img,0)是以灰度图的样式来读图片
    img2 = cv2.imread(img2)
    shape = img1.shape
    weidth, height = shape[0], shape[1]
    for i in weidth:
        for j in height:
            for k in range(3):
