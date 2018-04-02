import scipy.io
import numpy as np
import os
import scipy.misc
import scipy.stats as st

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt

x = [[1], [2], [3], [4], [5], [6]]
#x = [1, 2, 3, 4, 5, 6]
y = [1, 1, 1, 2, 1, 1]
x_y = zip(x, y)
def work(a, b):
    return (a +b)

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
    result = pool.map(work, [1, 2, 3], [1, 1, 1])
    pool.close()
    pool.join()
    print(result)"""
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
    for i in range(3):
        print(i)
