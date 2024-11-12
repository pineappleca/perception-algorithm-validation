#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:create_gt.py
@time:2024/11/11 16:45:28
@author:Yao Yongrui
'''

'''
生成与精简后的验证集相对应的真值数据结构并保存为.pkl文件
1. nuscenes_eval中的load_gt函数可以直接加载.pkl文件，从而减少评估所需时间
'''

import argparse
import copy
import json
import os
import time
from typing import Tuple, Dict, Any
import torch
import numpy as np
import pickle

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import NuScenesEval
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
import tqdm
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from torchvision.transforms.functional import rotate
import pycocotools.mask as mask_util
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from torchvision.transforms.functional import rotate
import cv2
import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, dist_pr_curve, visualize_sample
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.core.bbox.iou_calculators import BboxOverlaps3D
from IPython import embed
import json
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


class DetectionBox_modified(DetectionBox):
    def __init__(self, *args, token=None, visibility=None, index=None, **kwargs):
        '''
        add annotation token
        '''
        super().__init__(*args, **kwargs)
        self.token = token
        self.visibility = visibility
        self.index = index

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'token': self.token,
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
            'visibility': self.visibility,
            'index': self.index

        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(
            token=content['token'],
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name'],
            visibility=content['visibility'],
            index=content['index'],
        )


def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False):
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """

    # Init.
    if box_cls == DetectionBox_modified:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'
    index_map = {}
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        index_map[first_sample_token] = 1
        index = 2
        while sample['next'] != '':
            sample = nusc.get('sample', sample['next'])
            index_map[sample['token']] = index
            index += 1
    
    # 选取前5个验证场景作为验证集
    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split][:5]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox_modified:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        token=sample_annotation_token,
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name,
                        visibility=sample_annotation['visibility_token'],
                        index=index_map[sample_token]
                    )
                )
            elif box_cls == TrackingBox:
                assert False
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))
    
    # 将annotation加载到.pkl文件中
    output_filename = '/home/step/data/Documents/BEVFormer/data/nuscenes/nuscenes_partial_gt.pkl'

    with open(output_filename, 'wb') as f:
        pickle.dump(all_annotations, f)
    
    print(f"Ground truth annotations saved to {output_filename}")

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/step/data/Documents/BEVFormer/data/nuscenes', verbose=True)
    load_gt(nusc, 'val', DetectionBox_modified, verbose=True)