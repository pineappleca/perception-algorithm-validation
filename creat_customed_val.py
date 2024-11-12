#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:creat_customed_val.py
@time:2024/11/07 16:42:23
@author:Yao Yongrui
'''

'''
定制化生成nuscenes数据集的验证集
1. 可以缩小验证集，避免过长的等待时间或内存溢出问题；
2. 用于迭代测试
'''

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import pickle
# import os.path as osp

# 初始化 NuScenes 数据集
nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

# 获取验证集的场景
val_scenes = create_splits_scenes()['val']

# 选择你感兴趣的几个场景
selected_scenes = val_scenes[:5]  # 例如，只选择前5个场景

# 提取选定场景的信息
val_infos = []
for scene in nusc.scene:
    if scene['name'] in selected_scenes:
        sample_token = scene['first_sample_token']
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            sample_data = {
                'token': sample['token'],
                'timestamp': sample['timestamp'],
                'scene_token': sample['scene_token'],
                'data': sample['data'],
                'anns': sample['anns']
            }
            val_infos.append(sample_data)
            sample_token = sample['next']

# 将信息保存到 pkl 文件
with open('./data/nuscenes/partial_val.pkl', 'wb') as f:
    pickle.dump(val_infos, f)