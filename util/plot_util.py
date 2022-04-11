import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams.update({'font.size': 10})


def easy_plot(reals, predicts, title='compared predict and real'):
    """
    简易绘图方法
    :param title:
    :param reals:
    :param predicts:
    :return:
    """
    plt.figure()
    plt.plot(reals, color='b', label='real y')
    plt.plot(predicts, color='r', label='predict y')
    plt.title(title)  # 标题
    plt.legend()  # 自适应位置
    plt.show()
