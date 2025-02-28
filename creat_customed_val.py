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
import argparse
# import os.path as osp

# 解析参数
def parse_args():
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--split_all', action='store_true', default=False, help='whether generate .pkl file for every val scene')
    return parser.parse_args()

# 初始化 NuScenes 数据集
nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

# 获取验证集的场景
val_scenes = create_splits_scenes()['val']
args = parse_args()

# python creat_customed_val.py --split_all

# 只能通过遍历nusc.scene来获取所有的场景名
# 无法直接通过scene-0557获取nsc.scene
if args.split_all:
    # scene_name，例如：'scene-0557'
    gcount = 1
    for val_scene_name in val_scenes:
        # 提取选定场景的信息
        val_infos = []
        for scene in nusc.scene:
            if scene['name'] == val_scene_name:
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
                    # sample_token = sample['next']
                    sample_token = sample['next']

        # 将信息保存到 pkl 文件
        with open(f'./data/nuscenes/partial_val_{val_scene_name}.pkl', 'wb') as f:
            pickle.dump(val_infos, f)

        print(f"Process: {gcount}/{len(val_scenes)}. Save {len(val_infos)} samples to partial_val_{val_scene_name}.pkl")
        gcount += 1
    exit()

# 选择你感兴趣的几个场景
# selected_scenes = val_scenes[:5]  # 例如，只选择前5个场景
selected_scenes = ['scene-0557']

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

print(f"Save {len(val_infos)} samples to partial_val.pkl")