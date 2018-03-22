from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool
x = [[1], [2], [3], [4], [5], [6]]
#x = [1, 2, 3, 4, 5, 6]
y = [1, 1, 1, 1, 1, 1]
x_y = zip(x, y)
def work(a, b):
    return (len(a) +b)


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
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            print(i, j)
            for i in range(1, 3):
                x.append([2])
                y.append(1)
            p = ProcessingPool(processes=4)
            result = (p.map(work, x, y))
    p.close()
    p.join()

