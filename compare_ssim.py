from ssim import MultiScaleSSIM
import utils
import numpy as np
import cv2
PATCH_SIZE = 2592 * 1944 * 3
batch_size = 50
img1 = cv2.imread('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg')
img2 = cv2.imread('C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_cv2.imread.jpg')
flat_1 = np.reshape(img1, [-1, PATCH_SIZE])
flat_2 = np.reshape(img2, [-1, PATCH_SIZE])

loss_mse = np.sum(np.power(flat_2 - flat_1, 2))/PATCH_SIZE
loss_psnr = 20 * np.log10(1.0 / np.sqrt(loss_mse))

loss_ssim = MultiScaleSSIM(np.reshape(img1, [1,2592, 1944, 3]), np.reshape(img2, [1, 2592, 1944, 3]))
print(loss_psnr)
print(loss_ssim)

的广泛覆盖