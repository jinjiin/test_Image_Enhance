from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool
x = [1,2,3,4,5,6]
y = [1,1,1,1,1,1]
x_y = zip(x, y)
def work(a):
    return (a+8)


if __name__=="__main__":

    p = Pool(processes=4)
    result = (p.map(work, x))  #如何获取返回值
    p.close()
    p.join()
    print(result)
    """pool = ProcessingPool(nodes=4) # 如何传入多个参数
    pool.map(work, [1, 2, 3], [1, 1, 1])
    pool.close()
    pool.join()"""
