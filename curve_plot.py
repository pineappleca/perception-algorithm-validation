#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:curve_plot.py
@time:2024/11/27 13:55:01
@author:Yao Yongrui
'''

'''
绘制单失效因素条件下的曲线
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def single_corruption_index(corruption_name, title):
    # 读取 CSV 文件
    filename = f'{corruption_name}_result_eval.csv'
    data = pd.read_csv(filename)
    # 提取数据
    light_des = data[corruption_name]
    mAP = data['mAP']
    NDS = data['NDS']
    mAE = data['mAE']
    # 设置中文字体
    font = FontProperties(fname='/usr/share/fonts/chinses/simsun.ttc', size=18)
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.xlim(-5, 85)
    plt.ylim(-0.2, 1.2)

    # 绘制 mAP 曲线
    plt.plot(light_des, mAP, label='mAP', color='blue', linestyle='-', marker='o')
    # 绘制 NDS 曲线
    plt.plot(light_des, NDS, label='NDS', color='green', linestyle='--', marker='s')
    # 绘制 mAE 曲线
    plt.plot(light_des, mAE, label='mAE', color='red', linestyle='-.', marker='^')

    # 添加图例
    plt.legend(prop=font)

    # 设置标题和轴标签
    title_str = f'不同强度{title}失效条件下待测感知算法性能变化'
    plt.title(title_str, fontproperties=font)
    plt.xlabel('强度', fontproperties=font)
    plt.ylabel('性能指标值', fontproperties=font)
    # 显示图表
    plt.grid(True)
    plt.savefig('./plot/light_des_result_eval.png', dpi=600)

if __name__ == '__main__':
    single_corruption_index('light_des', '亮度降低')