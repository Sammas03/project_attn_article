from typing import Union, List, Dict
import pandas as pd
from util import dataframe_reader as dfr
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams.update({'font.size': 10})


def summary_data_plot(result_df_dict: Dict[str, List],title='predict result'):
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
    plt.figure()
    for index, (key, value) in enumerate(result_df_dict.items()):
        plt.plot(value, color=mcolors.TABLEAU_COLORS[colors[index]], label=key)
    plt.title(title)  # 标题
    plt.legend()  # 自适应位置
    plt.show()


def summary_data_plot_from_file(predict_col: Union[str, int], result_files: Dict[str, List]):
    link = {}
    reals = None
    for key, value in result_files.items():
        df = dfr.easy_read_data(value)
        link[key] = df[predict_col] if isinstance(predict_col, str) else df.iloc[:, predict_col]
        reals = df['real'].values
    link['real'] = reals
    summary_data_plot(link)
    # plot


def sins(flip=1):
    import numpy as np
    curves = {}
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        curves[i] = np.sin(x + i * .5) * (7 - i) * flip
    return curves


if __name__ == '__main__':
    # sns.set_palette('muted')
    curves = sins()
    summary_data_plot(curves)
