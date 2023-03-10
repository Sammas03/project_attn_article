'''
子序列的统计
'''


from typing import Union, List, Dict
import pandas as pd
import numpy as np
from util import dataframe_reader as dfr
import matplotlib.pyplot as plt
import seaborn as sns

import palettable
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams.update({'font.size': 12})



labels = ['L=6', 'L=9', 'L=12', 'L=24', 'L=36','L=72']

apt14= [4.42E-02,4.87E-02,5.73E-02,4.23E-02,4.64E-02,4.31E-02]
apt16 = [4.10E-02,6.86E-02,3.94E-02,3.67E-02,7.43E-02,3.97E-02]
apt101 = [5.13E-02,5.21E-02,5.18E-02,5.00E-02,5.33E-02,5.11E-02]

# sns.set(palette='muted')
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
plt.figure(figsize=(8,5))

rects1 = plt.bar(x - width/3, apt14, 0.2, label='APT14',color=palettable.matplotlib.Magma_8.mpl_colors[3])
rects2 = plt.bar(x , apt16, 0.2, label='APT16',color=palettable.matplotlib.Magma_8.mpl_colors[4])
rects3 = plt.bar(x + width/3, apt101, 0.2, label='APT101',color=palettable.matplotlib.Magma_8.mpl_colors[6])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('MAE')
plt.xlabel('Subsequence length')
# ax.set_title('Scores by group and gender')
plt.xticks(x,labels)
plt.legend()
#
# ax.bar_label(rects1, padding=2)
# ax.bar_label(rects2, padding=2)
# ax.bar_label(rects3, padding=2)

plt.savefig('/home/caowenzhi/lhd_attention/fig/子序列MAE条形图.svg')
plt.show()

