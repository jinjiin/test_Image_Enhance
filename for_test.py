import scipy.io
import numpy as np
import os
import scipy.misc
import scipy.stats as st

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
    nsig = 3
    kernlen = 21
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    A = np.arange(2, 14).reshape((1, 12))
    A[0,1] = 8
    A[0,4] = 20
    print(A, len(A[0]))
    print(np.diff(A), len(np.diff(A[0])))
    print(st.norm.cdf(x))
    print(kern1d)