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
from matplotlib.gridspec import GridSpec


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

def corruption_index_2(corruption_name_1, title1, corruption_name_2, title2):
    # 读取 CSV 文件
    filename_1 = f'{corruption_name_1}_result_eval_single.csv'
    data_corruption_1 = pd.read_csv(filename_1)
    filename_2 = f'{corruption_name_2}_result_eval_single.csv'
    data_corruption_2 = pd.read_csv(filename_2)
    
    # 提取数据
    corruption_severity_1 = data_corruption_1[corruption_name_1]
    # NOTE:车辆和行人的mAP
    mAP_1 = data_corruption_1['mAP']
    NDS_1 = data_corruption_1['NDS']
    mAE_1 = data_corruption_1['mAE']
    corruption_severity_2 = data_corruption_2[corruption_name_2]
    mAP_2 = data_corruption_2['mAP']
    NDS_2 = data_corruption_2['NDS']
    mAE_2 = data_corruption_2['mAE']

    # 创建图表
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].set_xlim(-5, 85)
    ax[0].set_ylim(-0.2, 1.5)
    ax[0].set_xticks(range(0, 85, 10))
    ax[0].set_yticks([0.25 * i for i in range(7)])
    ax[0].plot(corruption_severity_1, mAP_1, label='mAP', color='blue', linestyle='-', marker='o', markersize=8)
    ax[0].plot(corruption_severity_1, NDS_1, label='NDS', color='green', linestyle='--', marker='s', markersize=8)
    ax[0].plot(corruption_severity_1, mAE_1, label='mAE', color='red', linestyle='-.', marker='^', markersize=8)
    ax[0].legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 1.02))
    ax[0].set_title(f'不同强度{title1}条件下感知算法性能变化', fontsize=24, pad=20)
    ax[0].set_xlabel('强度', fontsize=20)
    ax[0].set_ylabel('性能指标值', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[0].grid(True)

    ax[1].set_xlim(-5, 85)
    ax[1].set_ylim(-0.2, 1.5)
    ax[1].set_xticks(range(0, 85, 10))
    ax[1].set_yticks([0.25 * i for i in range(7)])
    ax[1].plot(corruption_severity_2, mAP_2, label='mAP', color='blue', linestyle='-', marker='o', markersize=8)
    ax[1].plot(corruption_severity_2, NDS_2, label='NDS', color='green', linestyle='--', marker='s', markersize=8)
    ax[1].plot(corruption_severity_2, mAE_2, label='mAE', color='red', linestyle='-.', marker='^', markersize=8)
    ax[1].legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 1.02)) 
    ax[1].set_title(f'不同强度{title2}条件下感知算法性能变化', fontsize=24, pad=20)
    ax[1].set_xlabel('强度', fontsize=20)
    ax[1].set_ylabel('性能指标值', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    ax[1].grid(True)

    # plt.subplots_adjust(top=0.5)
    plt.tight_layout()
    plt.savefig(f'./plot/{title1}_{title2}_曲线图.svg', bbox_inches='tight')

def corruption_index_3(corruption_name_1, title1, corruption_name_2, title2, corruption_name_3, title3):
    # 读取 CSV 文件
    filename_1 = f'{corruption_name_1}_result_eval_single.csv'
    data_corruption_1 = pd.read_csv(filename_1)
    filename_2 = f'{corruption_name_2}_result_eval_single.csv'
    data_corruption_2 = pd.read_csv(filename_2)
    filename_3 = f'{corruption_name_3}_result_eval_single.csv'
    data_corruption_3 = pd.read_csv(filename_3)
    
    # 提取数据
    corruption_severity_1 = data_corruption_1[corruption_name_1]
    # NOTE:车辆和行人的mAP
    mAP_1 = data_corruption_1['mAP']
    NDS_1 = data_corruption_1['NDS']
    mAE_1 = data_corruption_1['mAE']
    corruption_severity_2 = data_corruption_2[corruption_name_2]
    mAP_2 = data_corruption_2['mAP']
    NDS_2 = data_corruption_2['NDS']
    mAE_2 = data_corruption_2['mAE']
    corruption_severity_3 = data_corruption_3[corruption_name_3]
    mAP_3 = data_corruption_3['mAP']
    NDS_3 = data_corruption_3['NDS']
    mAE_3 = data_corruption_3['mAE']

    # 创建图表
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.set_xlim(-5, 85)
    ax1.set_ylim(-0.2, 1.5)
    ax1.set_xticks(range(0, 85, 10))
    ax1.set_yticks([0.25 * i for i in range(7)])
    ax1.plot(corruption_severity_1, mAP_1, label='mAP', color='blue', linestyle='-', marker='o', markersize=8)
    ax1.plot(corruption_severity_1, NDS_1, label='NDS', color='green', linestyle='--', marker='s', markersize=8)
    ax1.plot(corruption_severity_1, mAE_1, label='mAE', color='red', linestyle='-.', marker='^', markersize=8)
    ax1.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 1.02))
    ax1.set_title(f'不同强度{title1}条件下感知算法性能变化', fontsize=24, pad=20)
    ax1.set_xlabel('强度', fontsize=20)
    ax1.set_ylabel('性能指标值', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True)

    ax2.set_xlim(-5, 85)
    ax2.set_ylim(-0.2, 1.5)
    ax2.set_xticks(range(0, 85, 10))
    ax2.set_yticks([0.25 * i for i in range(7)])
    ax2.plot(corruption_severity_2, mAP_2, label='mAP', color='blue', linestyle='-', marker='o', markersize=8)
    ax2.plot(corruption_severity_2, NDS_2, label='NDS', color='green', linestyle='--', marker='s', markersize=8)
    ax2.plot(corruption_severity_2, mAE_2, label='mAE', color='red', linestyle='-.', marker='^', markersize=8)
    ax2.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 1.02))
    ax2.set_title(f'不同强度{title2}条件下感知算法性能变化', fontsize=24, pad=20)
    ax2.set_xlabel('强度', fontsize=20)
    ax2.set_ylabel('性能指标值', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True)

    ax3.set_xlim(-5, 85)
    ax3.set_ylim(-0.2, 1.5)
    ax3.set_xticks(range(0, 85, 10))
    ax3.set_yticks([0.25 * i for i in range(7)])
    ax3.plot(corruption_severity_3, mAP_3, label='mAP', color='blue', linestyle='-', marker='o', markersize=8)
    ax3.plot(corruption_severity_3, NDS_3, label='NDS', color='green', linestyle='--', marker='s', markersize=8)
    ax3.plot(corruption_severity_3, mAE_3, label='mAE', color='red', linestyle='-.', marker='^', markersize=8)
    ax3.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 1.02))
    ax3.set_title(f'不同强度{title3}条件下感知算法性能变化', fontsize=24, pad=20)
    ax3.set_xlabel('强度', fontsize=20)
    ax3.set_ylabel('性能指标值', fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f'./plot/{title1}_{title2}_{title3}_曲线图.svg', bbox_inches='tight')


if __name__ == '__main__':
    # 设置中文字体为SimSun
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # 正常显示正负号
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制单失效因素条件下的曲线
    # corruption_index_2('light_aug', '光照增强', 'light_des', '光照减弱')
    # corruption_index_2('camera_blur', '镜头模糊', 'object_motion_sim', '目标模糊')
    corruption_index_3('add_rain', '降雨', 'add_snow', '降雪', 'add_fog', '浓雾')