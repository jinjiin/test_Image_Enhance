import scipy.io
import numpy as np
import os
import scipy.misc

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
x = [[1], [2], [3], [4], [5], [6]]
#x = [1, 2, 3, 4, 5, 6]
y = [1, 1, 1, 2, 1, 1]
x_y = zip(x, y)
def work(a, b):
    return (len(a) +b)

def work1(a):
    if a > 1:
        print("done")
        return 0
    print(a+5)
if __name__=="__main__":

    """p = ProcessingPool(processes=4)
    result = (p.map(work, x, y))  #如何获取返回值
    p.close()
    p.join()
    print(result)"""
    """pool = ProcessingPool(nodes=4) # 如何传入多个参数
    pool.map(work, [1, 2, 3], [1, 1, 1])
    pool.close()
    pool.join()"""
    """for i in range(1, len(x)):
        for j in range(1, len(y)):
            print(i, j)
            for i in range(1, 3):
                x.append([2])
                y.append(1)
            p = ProcessingPool(processes=4)
            result = (p.map(work, x, y))
    p.close()
    p.join()"""
    """p = mp.Pool(processes=10, maxtasksperchild=15)
    p.map_async(work1, y)
    p.close()
    p.join()"""
    VGG_PATH = "vgg_pretrained/imagenet-vgg-verydeep-19.mat"
    vgg = scipy.io.loadmat(VGG_PATH)
    print(type(vgg))
    print(vgg.keys())
    layers = vgg['layers']
    # print(layers)
    print(layers.shape)
    layer0 = layers[0]
    print(layer0.shape)
    print(layer0[0].shape)
    print(layer0[0][0].shape)
    print(layer0[0][0][0][0][0])