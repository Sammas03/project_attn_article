import os
import datetime


def easy_add_time_suffix(path: str):
    '''
    在文件路径的文件名后添加时间变量
    :param path:
    :return:
    '''

    sp_path = os.path.splitext(path)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return "{}-{}{}".format(sp_path[0], now, sp_path[1])


if __name__ == '__main__':
    print(easy_add_time_suffix('./ray/hello.txt'))
