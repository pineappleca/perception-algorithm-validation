import numpy as np
import torch
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image
import random



@PIPELINES.register_module()
class CropResizeFlipImage(object):
    """Fixed Crop and then randim resize and flip the image. Note the flip requires to flip the feature in the network   
    ida_aug_conf = {
        "reisze": [576, 608, 640, 672, 704]  # stride of 32 based on 640 (0.9, 1.1)
        "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768]  #  (0.8, 1.2)
        "reisze": [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832]  #  (0.7, 1.3)
        "crop": (0, 260, 1600, 900), 
        "H": 900,
        "W": 1600,
        "rand_flip": True,
}
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True, debug=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.debug = debug

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip = self._sample_augmentation(results)

        if self.debug:
            # unique id per img
            from uuid import uuid4
            uid = uuid4()
            # lidar is RFU in nuscenes
            lidar_pts = np.array([
                [10, 30, -2, 1],
                [-10, 30, -2, 1],
                [5, 15, -2, 1],
                [-5, 15, -2, 1],
                [30, 0, -2, 1],
                [-30, 0, -2, 1],
                [10, -30, -2, 1],
                [-10, -30, -2, 1]
            ], dtype=np.float32).T

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))

            if self.debug:
                pts_to_img_pre_aug = results['lidar2img'][i] @ lidar_pts
                pts_to_img_pre_aug = pts_to_img_pre_aug / pts_to_img_pre_aug[2:3,
                                                          :]  # div by the depth component in homogenous vector

                img_copy = Image.fromarray(np.uint8(imgs[i]))
                for j in range(pts_to_img_pre_aug.shape[1]):
                    x, y = int(pts_to_img_pre_aug[0, j]), int(pts_to_img_pre_aug[1, j])
                    if (0 < x < img_copy.width) and (0 < y < img_copy.height):
                        img_copy.putpixel((x - 1, y - 1), (255, 0, 0))
                        img_copy.putpixel((x - 1, y), (255, 0, 0))
                        img_copy.putpixel((x - 1, y + 1), (255, 0, 0))
                        img_copy.putpixel((x, y - 1), (0, 255, 0))
                        img_copy.putpixel((x, y), (0, 255, 0))
                        img_copy.putpixel((x, y + 1), (0, 255, 0))
                        img_copy.putpixel((x + 1, y - 1), (0, 0, 255))
                        img_copy.putpixel((x + 1, y), (0, 0, 255))
                        img_copy.putpixel((x + 1, y + 1), (0, 0, 255))
                img_copy.save(f'pre_aug_{uid}_{i}.png')

            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][i][:3, :3] = np.matmul(ida_mat, results['cam2img'][i][:3, :3])

            if self.debug:
                pts_to_img_post_aug = np.matmul(results['cam2img'][i], results['lidar2cam'][i]) @ lidar_pts
                pts_to_img_post_aug = pts_to_img_post_aug / pts_to_img_post_aug[2:3,
                                                            :]  # div by the depth component in homogenous vector
                for j in range(pts_to_img_post_aug.shape[1]):
                    x, y = int(pts_to_img_post_aug[0, j]), int(pts_to_img_post_aug[1, j])
                    if (0 < x < img.width) and (0 < y < img.height):
                        img.putpixel((x - 1, y - 1), (255, 0, 0))
                        img.putpixel((x - 1, y), (255, 0, 0))
                        img.putpixel((x - 1, y + 1), (255, 0, 0))
                        img.putpixel((x, y - 1), (0, 255, 0))
                        img.putpixel((x, y), (0, 255, 0))
                        img.putpixel((x, y + 1), (0, 255, 0))
                        img.putpixel((x + 1, y - 1), (0, 0, 255))
                        img.putpixel((x + 1, y), (0, 0, 255))
                        img.putpixel((x + 1, y + 1), (0, 0, 255))
                img.save(f'post_aug_{uid}_{i}.png')

            if 'mono_ann_idx' in results.keys():
                # apply transform to dd3d intrinsics
                if i in results['mono_ann_idx'].data:
                    mono_index = results['mono_ann_idx'].data.index(i)
                    intrinsics = results['mono_input_dict'][mono_index]['intrinsics']
                    if torch.is_tensor(intrinsics):
                        intrinsics = intrinsics.numpy().reshape(3, 3).astype(np.float32)
                    elif isinstance(intrinsics, np.ndarray):
                        intrinsics = intrinsics.reshape(3, 3).astype(np.float32)
                    else:
                        intrinsics = np.array(intrinsics, dtype=np.float32).reshape(3, 3)
                    results['mono_input_dict'][mono_index]['intrinsics'] = np.matmul(ida_mat, intrinsics)
                    results['mono_input_dict'][mono_index]['height'] = img.size[1]
                    results['mono_input_dict'][mono_index]['width'] = img.size[0]

                    # apply transform to dd3d box
                    for ann in results['mono_input_dict'][mono_index]['annotations']:
                        # bbox_mode = BoxMode.XYXY_ABS
                        box = self._box_transform(ann['bbox'], resize, crop, flip, img.size[0])[0]
                        box = box.clip(min=0)
                        box = np.minimum(box, list(img.size + img.size))
                        ann["bbox"] = box

        results["img"] = new_imgs
        results['lidar2img'] = [np.matmul(results['cam2img'][i], results['lidar2cam'][i]) for i in
                                range(len(results['lidar2cam']))]

        return results

    def _box_transform(self, box, resize, crop, flip, img_width):
        box = np.array([box])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)

        # crop
        coords[:, 0] -= crop[0]
        coords[:, 1] -= crop[1]

        # resize
        coords[:, 0] = coords[:, 0] * resize
        coords[:, 1] = coords[:, 1] * resize

        coords = coords.reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_box = np.concatenate((minxy, maxxy), axis=1)

        return trans_box

    def _img_transform(self, img, resize, resize_dims, crop, flip):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.crop(crop)
        img = img.resize(resize_dims)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2]) * resize
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, results):
        if 'CropResizeFlipImage_param' in results['aug_param'].keys():
            return results['aug_param']['CropResizeFlipImage_param']
        crop = self.data_aug_conf["crop"]

        if self.training:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
        else:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            assert len(self.data_aug_conf["reisze"]) == 1
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
        results['aug_param']['CropResizeFlipImage_param'] = (resize, resize_dims, crop, flip)

        return resize, resize_dims, crop, flip


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
            self,
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
            reverse_angle=False,
            training=True,
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            only_gt=False,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

        self.flip_dx_ratio = flip_dx_ratio
        self.flip_dy_ratio = flip_dy_ratio
        self.only_gt = only_gt

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}

        rot_angle, scale_ratio, flip_dx, flip_dy, _, _ = self._sample_augmentation(results)

        # random rotate
        if not self.only_gt:
            self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )

        # random scale
        if not self.only_gt:
            self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # random flip
        if flip_dx:
            if not self.only_gt:
                self.flip_along_x(results)
            results["gt_bboxes_3d"].flip(bev_direction='vertical')
        if flip_dy:
            if not self.only_gt:
                self.flip_along_y(results)
            results["gt_bboxes_3d"].flip(bev_direction='horizontal')

        # TODO: support translation
        return results

    def _sample_augmentation(self, results):
        if 'GlobalRotScaleTransImage_param' in results['aug_param'].keys():
            return results['aug_param']['GlobalRotScaleTransImage_param']
        else:
            rot_angle = np.random.uniform(*self.rot_range) / 180 * np.pi
            scale_ratio = np.random.uniform(*self.scale_ratio_range)
            flip_dx = np.random.uniform() < self.flip_dx_ratio
            flip_dy = np.random.uniform() < self.flip_dy_ratio
        # generate bda_mat 

        rot_sin = torch.sin(torch.tensor(rot_angle))
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        bda_mat = flip_mat @ (scale_mat @ rot_mat)
        bda_mat = torch.inverse(bda_mat)
        results['aug_param']['GlobalRotScaleTransImage_param'] = (
        rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt)

        return rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], rot_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], rot_mat_inv)

        return

    def scale_xyz(self, results, scale_ratio):
        scale_mat = np.array(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = np.linalg.inv(scale_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], scale_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], scale_mat_inv)
        return

    def flip_along_x(self, results):
        flip_mat = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], flip_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], flip_mat_inv)
        return

    def flip_along_y(self, results):
        flip_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], flip_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], flip_mat_inv)
        return

from .Camera_corruptions import ImagePointAddSun, ImageAddSunMono, ImageLightAug, ImageBBoxMotionBlurFrontBack, ImageBBoxMotionBlurLeftRight, ImageBBoxMotionBlurFrontBackMono, ImageBBoxMotionBlurLeftRightMono

# 模拟失效触发条件
@PIPELINES.register_module()
class CorruptionMethods(object):
    """
    Test-time augmentation with corruptions.

    取值范围--[0, 80]

    flare_aug--光照增强
    sun_sim--局部光斑
    """

    def __init__(self,
                 corruption_severity_dict=
                    {
                        'sun_sim':20,
                        'light_aug':20,
                    },
                 ):
                 
        self.corruption_severity_dict = corruption_severity_dict
        
        if 'sun_sim' in self.corruption_severity_dict:
            np.random.seed(2022)
            severity = self.corruption_severity_dict['sun_sim']
            # self.sun_sim = ImagePointAddSun(severity)
            self.sun_sim_mono = ImageAddSunMono(severity)
        
        if 'light_aug' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['light_aug']
            self.light_aug = ImageLightAug(severity)
        
         #for multiple cameras corruption
         
        # if 'object_motion_sim' in self.corruption_severity_dict:
        #     # for kitti and nus
        #     severity = self.corruption_severity_dict['object_motion_sim']
        #     self.object_motion_sim_frontback = ImageBBoxMotionBlurFrontBack(
        #         severity=severity,
        #         corrput_list=[0.02 * i for i in range(1, 6)],
        #     )
        #     self.object_motion_sim_leftright = ImageBBoxMotionBlurLeftRight(
        #         severity=severity,
        #         corrput_list=[0.02 * i for i in range(1, 6)],
        #     )
        #     self.object_motion_sim_frontback_mono = ImageBBoxMotionBlurFrontBackMono(
        #         severity=severity,
        #         corrput_list=[0.02 * i for i in range(1, 6)],
        #     )
        #     self.object_motion_sim_leftright_mono = ImageBBoxMotionBlurLeftRightMono(
        #         severity=severity,
        #         corrput_list=[0.02 * i for i in range(1, 6)],
        #     )
            
            

    def __call__(self, results):
        """Call function to augment common corruptions.
        """

        img_bgr_255_np_uint8 = results['img']  # nus:list / kitti: nparray
        image_aug_rgb = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
        
        # 运动模糊
        # if 'montion_blur' in self.corruption_severity_dict:
        #     image_aug_rgb = self.montion_blur(
        #         image=image_aug_rgb,
        #     )
        
        # 局部光斑(仅对前置摄像头生效)
        if 'sun_sim' in self.corruption_severity_dict:
            # image_aug_rgb = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
            image_aug_rgb = self.sun_sim_mono(
                image=image_aug_rgb,
            )
        # 生成随机数
        
        
        # 整体亮度增强(对所有摄像头生效)
        if 'light_aug' in self.corruption_severity_dict:
            image_aug_rgb = self.light_aug(
                image=image_aug_rgb,
            )
        
        image_aug_bgr_0 = image_aug_rgb[:, :, [2, 1, 0]]
        img_bgr_255_np_uint8[0] = image_aug_bgr_0
        results['img'] = img_bgr_255_np_uint8
        return results

            # 尝试去掉雷达数据
            # print(results.keys())
            # raise AssertionError('stop here')

            # if 'points' not in results:
            #     use_mono_dataset = True
            # elif 'lidar2img' in results:
            #     lidar2img = results['lidar2img']  # nus: list / kitti: nparray
            #     use_mono_dataset = False

            # if 'lidar2img' in results:
            #     lidar2img = results['lidar2img']  # nus: list / kitti: nparray
            #     use_mono_dataset = True
            # if 'cam_intrinsic' in results['img_info']:
            #     cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono dataset
            #     import numpy as np
            #     cam2img = np.array(cam2img)
            #     use_mono_dataset = True
            # else:
            #     raise AssertionError('no lidar2img or cam_intrinsic found!')

            # if not use_mono_dataset:
            #     points_tensor = results['points'].tensor
            #     # different part of nus and kitti
            #     if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
            #         # nus dataset
            #         # only one sun
            #         '''
            #         nuscenes:
            #         0    CAM_FRONT,
            #         1    CAM_FRONT_RIGHT,
            #         2    CAM_FRONT_LEFT,
            #         3    CAM_BACK,
            #         4    CAM_BACK_LEFT,
            #         5    CAM_BACK_RIGHT
            #         '''
            #         img_rgb_255_np_uint8_0 = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
            #         lidar2img_0 = lidar2img[0]
            #         image_aug_rgb_0, points_aug = self.sun_sim(
            #             image=img_rgb_255_np_uint8_0,
            #             points=points_tensor,
            #             lidar2img=lidar2img_0,
            #         )
            #         image_aug_bgr_0 = image_aug_rgb_0[:, :, [2, 1, 0]]
            #         img_bgr_255_np_uint8[0] = image_aug_bgr_0
            #         results['img'] = img_bgr_255_np_uint8
            #         results['points'].tensor = points_aug

            #     elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
            #         # waymo dataset
            #         # only one sun
            #         '''
            #         nuscenes:
            #         0    CAM_FRONT,
            #         1    CAM_FRONT_RIGHT,
            #         2    CAM_FRONT_LEFT,
            #         3    CAM_BACK,
            #         4    CAM_BACK_LEFT,
            #         5    CAM_BACK_RIGHT
            #         '''
            #         img_rgb_255_np_uint8_0 = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
            #         lidar2img_0 = lidar2img[0]
            #         image_aug_rgb_0, points_aug = self.sun_sim(
            #             image=img_rgb_255_np_uint8_0,
            #             points=points_tensor,
            #             lidar2img=lidar2img_0,
            #         )
            #         image_aug_bgr_0 = image_aug_rgb_0[:, :, [2, 1, 0]]
            #         img_bgr_255_np_uint8[0] = image_aug_bgr_0
            #         results['img'] = img_bgr_255_np_uint8
            #         results['points'].tensor = points_aug

            #     else:
            #         # kitti dataset
            #         img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
            #         image_aug_rgb, points_aug = self.sun_sim(
            #             image=img_rgb_255_np_uint8,
            #             points=points_tensor,
            #             lidar2img=lidar2img,
            #         )
            #         image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
            #         results['img'] = image_aug_bgr
            #         results['points'].tensor = points_aug
            # else:
            #     # mono dataset of nus and kitti only one image
            #     # 只对前置摄像头进行调整
            #     img_rgb_255_np_uint8_0 = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
            #     image_aug_rgb = self.sun_sim_mono(
            #         image=img_rgb_255_np_uint8_0,
            #     )
            #     image_aug_bgr_0 = image_aug_rgb[:, :, [2, 1, 0]]
            #     img_bgr_255_np_uint8[0] = image_aug_bgr_0
            #     print(type(img_bgr_255_np_uint8))
            #     print(np.array(img_bgr_255_np_uint8).shape)
            #     results['img'] = img_bgr_255_np_uint8

        # if 'object_motion_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     # points_tensor = results['points'].tensor
        #     if 'lidar2img' in results:
        #         lidar2img = results['lidar2img']  
        #         use_mono_dataset = False
        #     elif 'cam_intrinsic' in results['img_info']:
        #         cam2img = results['img_info']['cam_intrinsic']  
        #         import numpy as np
        #         cam2img = np.array(cam2img)
        #         use_mono_dataset = True
        #     else:
        #         raise AssertionError('no lidar2img or cam_intrinsic found!')
            
        #     bboxes_corners = results['gt_bboxes_3d'].corners
        #     bboxes_centers = results['gt_bboxes_3d'].center


        #     if type(bboxes_corners) == int:
        #         print(0)

        #     if type(bboxes_corners) != int:

        #         if not use_mono_dataset:
        #             if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
        #                 '''
        #                 nuscenes:
        #                 0    CAM_FRONT,
        #                 1    CAM_FRONT_RIGHT,
        #                 2    CAM_FRONT_LEFT,
        #                 3    CAM_BACK,
        #                 4    CAM_BACK_LEFT,
        #                 5    CAM_BACK_RIGHT
        #                 '''
        #                 image_aug_bgr = []
        #                 for i in range(6):
        #                     img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
        #                     lidar2img_i = lidar2img[i]

        #                     if i % 3 == 0:
        #                         image_aug_rgb_i = self.object_motion_sim_frontback(
        #                             image=img_rgb_255_np_uint8_i,
        #                             bboxes_centers=bboxes_centers,
        #                             bboxes_corners=bboxes_corners,
        #                             lidar2img=lidar2img_i,
        #                             # watch_img=True,
        #                             # file_path='2.jpg'
        #                         )
        #                     else:
        #                         image_aug_rgb_i = self.object_motion_sim_leftright(
        #                             image=img_rgb_255_np_uint8_i,
        #                             bboxes_centers=bboxes_centers,
        #                             bboxes_corners=bboxes_corners,
        #                             lidar2img=lidar2img_i,
        #                             # watch_img=True,
        #                             # file_path='2.jpg'
        #                         )
        #                         # print('object_motion_sim_leftright:', time_inter)
        #                     image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
        #                     image_aug_bgr.append(image_aug_bgr_i)
        #                 results['img'] = image_aug_bgr

        #             elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
        #                 '''
        #                 nuscenes:
        #                 0    CAM_FRONT,
        #                 1    CAM_FRONT_RIGHT,
        #                 2    CAM_FRONT_LEFT,
        #                 3    CAM_BACK,
        #                 4    CAM_BACK_LEFT,
        #                 5    CAM_BACK_RIGHT
        #                 '''
        #                 image_aug_bgr = []
        #                 for i in range(5):
        #                     img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
        #                     lidar2img_i = lidar2img[i]

        #                     # if i % 3 == 0:
        #                     if i == 0:
        #                         image_aug_rgb_i = self.object_motion_sim_frontback(
        #                             image=img_rgb_255_np_uint8_i,
        #                             bboxes_centers=bboxes_centers,
        #                             bboxes_corners=bboxes_corners,
        #                             lidar2img=lidar2img_i,
        #                             # watch_img=True,
        #                             # file_path='2.jpg'
        #                         )
        #                     else:
        #                         image_aug_rgb_i = self.object_motion_sim_leftright(
        #                             image=img_rgb_255_np_uint8_i,
        #                             bboxes_centers=bboxes_centers,
        #                             bboxes_corners=bboxes_corners,
        #                             lidar2img=lidar2img_i,
        #                             # watch_img=True,
        #                             # file_path='2.jpg'
        #                         )
        #                         # print('object_motion_sim_leftright:', time_inter)
        #                     image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
        #                     image_aug_bgr.append(image_aug_bgr_i)
        #                 results['img'] = image_aug_bgr

        #             else:
        #                 img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
        #                 # points_tensor = results['points'].tensor
        #                 lidar2img = results['lidar2img']
        #                 bboxes_corners = results['gt_bboxes_3d'].corners
        #                 bboxes_centers = results['gt_bboxes_3d'].center

        #                 image_aug_rgb = self.object_motion_sim_frontback(
        #                     image=img_rgb_255_np_uint8,
        #                     bboxes_centers=bboxes_centers,
        #                     bboxes_corners=bboxes_corners,
        #                     lidar2img=lidar2img,
        #                     # watch_img=True,
        #                     # file_path='2.jpg'
        #                 )
        #                 image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
        #                 results['img'] = image_aug_bgr

        #         else:
        #             img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
        #             image_aug_rgb = self.object_motion_sim_frontback_mono(
        #                 image=img_rgb_255_np_uint8,
        #                 bboxes_centers=bboxes_centers,
        #                 bboxes_corners=bboxes_corners,
        #                 cam2img=cam2img,
        #                 # watch_img=True,
        #                 # file_path='2.jpg'
        #             )
        #             image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
        #             results['img'] = image_aug_bgr
