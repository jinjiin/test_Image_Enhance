import cv2
from matplotlib import pyplot as plt
import numpy as np

h = 1536
w = 2048
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
print(pts)
img2 = cv2.imread('test_image\\original_images\\canon\\85.jpg')
cv2.rectangle(img2, (3263, 2079), (1051, 9), (255, 255, 0), 3)
cv2.circle(img2, (3263, 2079), 100, (0, 255, 0))
cv2.circle(img2, (1051, 9), 100, (255, 0, 0))
cv2.imwrite("test_rec.jpg", img2)

img2 = img2[9:2079, int(1051):3263]
cv2.imwrite("test_cut.jpg", img2)
print(img2.shape)

img3 = cv2.resize(img2, (int(img2.shape[0]/2), int(img2.shape[1]/2)), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("test_resize.jpg", img3)

