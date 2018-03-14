import multiprocessing as mp

x = [1,2,3,4,5,6]
y = [1,1,1,1,1,1]
x_y = zip(x, y)
def work(a):
    print(a+8)


if __name__=="__main__":
    p = mp.Pool(processes=4)
    p.map_async(work, x)
    p.close()
    p.join()
