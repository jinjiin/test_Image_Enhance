import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('test_image\\original_images\\iphone\\85.jpg')


cv2.rectangle(img2, (500, 500), (800, 800), (255, 0, 0), 1, 8, 0)
cv2.imwrite("test_rec.jpg", img2)
img2 = img2[500:800, 500:800]
cv2.imwrite("test_cut.jpg", img2)

plt.imshow(img2)
plt.show()