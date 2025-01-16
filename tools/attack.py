#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:attack.py
@time:2024/12/21 19:57:58
@author:Yao Yongrui
'''

'''
使用对抗性攻击算法，生成对抗性测试用例
'''

import subprocess
import json
import os
import numpy as np
import torch
import math
import csv

# 配置文件路径
config_path = "./projects/configs/bevformer/bevformer_base.py"
# 检查点文件路径
checkpoint_path = "./ckpts/bevformer_r101_dcn_24ep.pth"
# 其他固定参数
# launcher = "pytorch"
eval_metric = "bbox"

def get_f_value(a, b, c, config_path, checkpoint_path, eval_metric):
    '''
    运行模型获取f(x)
    '''
    command = [
        "python", "./tools/test.py", 
        config_path, 
        checkpoint_path, 
        "--eval", eval_metric,
        "--corruption", json.dumps({"add_rain": a, "sensor_gnoise": b, "camera_blur": c})
    ]

    # 打印命令（可选）
    print("Executing command:", " ".join(command))
    # 获取上一级目录的绝对路径
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 在PYTHONPATH下执行命令
    # result = subprocess.run(command, env={**os.environ, 'PYTHONPATH': parent_dir}, capture_output=True, text=True)
    result = subprocess.run(command, env={**os.environ, 'PYTHONPATH': parent_dir}, capture_output=True, text=True)
    # 删除corruption_valid文件夹
    # delete_command = "rm -r corruption_valid/*"
    # subprocess.run(delete_command, shell=True)
    output = result.stdout
    # return pow(a, 2) + 1 / b + math.exp(c)
    # print(output)
    return float(output.strip().split('\n')[-2])

def pgd_attack(xs_t, grad_estimates, lr, p, eps):
    '''
    PGD攻击
    xs_t: torch.tensor, shape:(3,)
    grad_estimates: torch.tensor, shape:(3,)
    lr: float
    p: int
    eps: float, epsilon，球体半径
    output: torch.tensor, shape:(3,)
    '''
    # l2-ball
    if p == '2':
        delta = lr * grad_estimates / torch.norm(grad_estimates, p=2)
        # print(f'delta: {delta}, norm: {torch.norm(delta, p=2)}')
        if torch.norm(delta, p=2) > eps:
            delta = delta * (eps / torch.norm(delta, p=2))
        return xs_t + delta
    
    # lp-ball
    elif p == 'inf':
        delta = lr * torch.sign(grad_estimates)
        # 将delta限制在[-eps, eps]之间
        delta = torch.clamp(delta, -eps, eps)
        return xs_t + delta


def write_res_tocsv(algorithm_name, query_time, grad_estimates_list, f_attack_output):
    '''
    将梯度估计值和mAP写入文件
    algorithm_name: 算法名称, str, 'nes_pgd'
    '''
    if not os.path.exists('./attack_output'):
        os.mkdir('./attack_output')
    
    # 写入列名
    attack_csv_path = f'./attack_output/{algorithm_name}.csv'
    if not os.path.exists(attack_csv_path):
        with open(attack_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['query_time', 'grad_estimate_1', 'grad_estimate_2', 'grad_estimate_3', 'f_attack'])
    
    # 写入数据
    with open(attack_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([query_time] + grad_estimates_list + [f_attack_output])
    

def reward_function(f_value, f0):
    '''
    计算reward
    f_value: torch.tensor, shape:(N,)
    f0: float
    output: torch.tensor, shape:(N,)
    '''
    return 0.8 - f_value / f0


def NES_grad(initial_params, N, sigma=0.1, lr=0.1, p='inf', max_iters=1001, eps=0.1):
    '''
    NES生成算法
    initial_params: 初始参数，如[5, 5, 5]
    N：种群数量，int
    sigma:搜索方差，float
    '''

    a0, b0, c0 = initial_params
    # 初始参数，用于PGD控制步长
    xs_0 = torch.tensor([a0, b0, c0], dtype=torch.float32)
    f0 = get_f_value(a0, b0, c0, config_path, checkpoint_path, eval_metric)
    xs_t = xs_0
    # 访问模型的次数
    query_time = 1
    # 初始化梯度值
    grad_estimates = np.zeros(3)
    print(f'query time: {query_time}, gradient estimate: {grad_estimates}, mAP value: {f0}, parameters: {xs_t}')
    f_attack = f0

    # 终止条件修改为目标函数>=0或达到最大迭代次数
    while reward_function(f_attack, f0) < 0 and query_time < max_iters:
        noise = torch.randn(N, 3)
        fxs_t = xs_t + sigma * noise
        bxs_t = xs_t - sigma * noise

        # 保证参数范围
        fxs_t = torch.clamp(fxs_t, min=0, max=80)
        bxs_t = torch.clamp(bxs_t, min=0, max=80)
        fxst_values = torch.tensor([get_f_value(fxs_t[i, 0].item(), fxs_t[i, 1].item(), fxs_t[i, 2].item(), config_path, checkpoint_path, eval_metric) for i in range(N)])
        bxst_values = torch.tensor([get_f_value(bxs_t[i, 0].item(), bxs_t[i, 1].item(), bxs_t[i, 2].item(), config_path, checkpoint_path, eval_metric) for i in range(N)])

        rewards = reward_function(fxst_values, f0) - reward_function(bxst_values, f0) # shape:(N,)
        # print(rewards.shape, noise.shape)
        gs_ls = torch.mul(rewards.unsqueeze(1), noise) / (2 * sigma)
        grad_estimates = torch.mean(gs_ls, dim=0)
        # grad_estimates_output = torch.round(grad_estimates * 100) / 100

        # PGD更新参数
        xs_t = pgd_attack(xs_t, grad_estimates, lr, p, eps)
        # xs_t_output = torch.round(xs_t * 100) / 100  # 保留两位小数
        f_attack = get_f_value(xs_t[0].item(), xs_t[1].item(), xs_t[2].item(), config_path, checkpoint_path, eval_metric)
        f_attack_output = round(f_attack, 4)
        query_time += N

        # 将梯度估计值和mAP写入文件
        write_res_tocsv('nes_pgd', query_time, [round(grad_estimates[0].item(), 4), round(grad_estimates[1].item(), 4), round(grad_estimates[2].item(), 4)], f_attack_output)
        
        # TODO:绘图
        print(f'query time: {query_time}, gradient estimate: {grad_estimates}, mAP value: {f_attack_output}, parameters: {xs_t}')

NES_grad([5, 5, 5], 10)
