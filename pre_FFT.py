import cv2
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft,ifft
import numpy as np
def compare(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    shape = img1.shape
    weidth, height = shape[0], shape[1]
    print(weidth, height)
    diff = 0
    for i in range(weidth):
        for j in range(height):
            for k in range(3):
                if not img1[i][j][k] == img2[i][j][k]:
                    diff = diff + 1
                    print((i, j, k), img1[i][j][k], img2[i][j][k])
    print(diff)
def FFT(img):
    # 打开图像文件并获取数据
    srcIm = Image.open(img)
    print(srcIm) # PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2592x1944 at 0xB39CE1EC18
    srcArray = np.fromstring(srcIm.tobytes(), dtype=np.int8)
    print(type(srcArray))  # <class 'numpy.ndarray'>
    # 傅里叶变换并滤除低频信号
    result = fft(srcIm)  # result.shape=(1944, 2592, 3)
    # result = np.where(np.absolute(result) < 9e4, 0, result)
    # 傅里叶反变换,保留实部
    result = ifft(result)  # result.shape=(1944, 2592, 3)
    result = np.int8(np.real(result))
    # 转换为图像
    """im = Image.frombytes(srcIm.mode, srcIm.size, result)
    im.show()
    im.save('C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_2.jpg')"""
if __name__ == '__main__':
    FFT('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg')
    #compare('C:\\Users\\chenjinjin\\Desktop\\test_image\\50.jpg', 'C:\\Users\\chenjinjin\\Desktop\\test_image\\50_ufft_2.jpg')