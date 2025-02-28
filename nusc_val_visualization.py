#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:nusc_val_visualization.py
@time:2024/11/17 10:21:35
@author:Yao Yongrui
'''

'''
可视化nuscenes数据集的验证集，辅助提取特征，筛选测试场景
'''

import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

# 初始化 NuScenes 数据集
nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
output_dir = './nuscenes_video_train'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def list_scenes_by_names(scene_names):
    """
    列出给定场景名称列表中所有场景的信息。

    参数:
    scene_names (list): 场景名称列表，例如 ['scene-001', 'scene-002', ...]
    """
    for scene in nusc.scene:
        if scene['name'] in scene_names:
            print(f"Scene name: {scene['name']}", end=' ')
            print(f"Scene token: {scene['token']}", end=' ')
            print(f"Description: {scene['description']}", end=' ')
            print(f"Number of samples: {scene['nbr_samples']}", end=' ')
            print(f"First sample token: {scene['first_sample_token']}", end=' ')
            print(f"Last sample token: {scene['last_sample_token']}", end=' ')
            # print()

def generate_video(scene_names, output_dir):
    for scene_name in tqdm(scene_names, desc="Generating videos"):
        table_name, field, query = 'scene', 'name', scene_name
        my_scene_token = nusc.field2token(table_name, field, query)[0]
        # print("token", my_scene_token)
        nusc.render_scene_channel(my_scene_token, 'CAM_FRONT', out_path=os.path.join(output_dir, f"{query}.avi"))

# 示例场景名称列表
val_scene_names = create_splits_scenes()['train']
print(val_scene_names)

# 列出给定场景名称列表中所有场景的信息
list_scenes_by_names(val_scene_names)
generate_video(val_scene_names, output_dir)