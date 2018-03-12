import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('test_image\\original_images\\iphone\\85.jpg')
cv2.rectangle(img2, (500, 500), (800, 800), (255, 255, 0), 3)
cv2.imwrite("test_rec.jpg", img2)

img2 = img2[500:800, 500:800]
cv2.imwrite("test_cut.jpg", img2)

img3 = cv2.resize(img2, (100, 100), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("test_resize.jpg", img3)
plt.imshow(img2)
plt.show()