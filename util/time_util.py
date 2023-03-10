import time
import time as t

class TimeUtil():
    def __init__(self):
        self.reset()

    def count(self,show=False):
        spend = time.time() - self.start
        if(show):
            print("运行时间:%.2f秒"%spend)
        return spend

    def reset(self):
        self.start = time.time()


if __name__ == '__main__':
    tcount = TimeUtil()
    for _ in range(100000000):
        pass
    tcount.count(True)