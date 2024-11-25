#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:iteration_test.py
@time:2024/11/25 16:43:27
@author:Yao Yongrui
'''

'''
控制bevformer进行迭代测试
'''

import subprocess
import json
import os

# 定义 corruption 参数的键和值
# corruption_params = {
#     "light_des": [10, 20, 30],
#     "object_motion_sim": [2, 4, 6],
#     "sun_sim": [60, 80, 100]
# }

# corruption_params = {
#     "light_des": list(range(0, 81, 4)),
# }

corruption_params = {
    "light_des": [0, 10],
}

# 配置文件路径
config_path = "./projects/configs/bevformer/bevformer_base.py"
# 检查点文件路径
checkpoint_path = "./ckpts/bevformer_r101_dcn_24ep.pth"
# 其他固定参数
# launcher = "pytorch"
eval_metric = "bbox"

# 遍历 corruption 参数的键和值
def generate_corruption_dicts(corruption_params):
    keys = list(corruption_params.keys())
    values = list(corruption_params.values())
    for combination in product(*values):
        yield dict(zip(keys, combination))

from itertools import product

for corruption_dict in generate_corruption_dicts(corruption_params):
    # 将 corruption_dict 转换为 JSON 字符串
    corruption_str = json.dumps(corruption_dict)
    
    # 构建命令
    command = [
        "python", "./tools/test.py",
        config_path,
        checkpoint_path,
        # "--launcher", launcher,
        "--eval", eval_metric,
        "--corruption", corruption_str
    ]
    
    # 打印命令（可选）
    print("Executing command:", " ".join(command))

    # 获取上一级目录的绝对路径
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 在PYTHONPATH下执行命令
    subprocess.run(command, env={**os.environ, 'PYTHONPATH': parent_dir})
    
    # 删除 corruption_valid 目录下的所有文件
    delete_command = "rm -r corruption_valid/*"
    subprocess.run(delete_command, shell=True)