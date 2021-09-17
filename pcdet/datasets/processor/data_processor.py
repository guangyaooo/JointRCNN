from functools import partial

import numpy as np

from ...utils import box_utils, common_utils
from ..kitti.kitti_utils import pad_image, get_fov_flag
from skimage import transform as st
import copy
from easydict import EasyDict


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=len(choice)*2 < num_points)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=len(choice)*2 < num_points)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


class FusionProcessor(DataProcessor):
    def __init__(self, processor_configs, point_cloud_range, training):
        super().__init__(processor_configs, point_cloud_range, training)

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        assert False, 'FusionProcessor is not supported to transform points to voxels'

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        # FOV_POINTS_ONLY:
        calib, points = data_dict['calib'], data_dict['points']
        pts_rect = data_dict['calib'].lidar_to_rect(points[:, 0:3])
        fov_flag, pts_img = get_fov_flag(pts_rect, data_dict['image_shape'],
                                         calib, True)

        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        mask = np.logical_and(mask, fov_flag)
        data_dict['points'] = data_dict['points'][mask]
        data_dict['pts_img'] = pts_img[mask]
        data_dict['fov_indices'] = data_dict['fov_indices'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            # TODO Masking a target box whether make the network confusing ?
            data_dict['gt_boxes_2d'] = data_dict['gt_boxes_2d'][mask]
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=len(choice)*2 < num_points)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=len(choice)*2 < num_points)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        data_dict['pts_img'] = data_dict['pts_img'][choice]
        data_dict['fov_indices'] = data_dict['fov_indices'][choice]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points
            data_dict['pts_img'] = data_dict['pts_img'][shuffle_idx]
            data_dict['fov_indices'] = data_dict['fov_indices'][shuffle_idx]

        return data_dict

    def resize_image(self, data_dict=None, config=None):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        if data_dict is None:
            return partial(self.resize_image, config=config)

        img, boxes, calib = data_dict['images'], data_dict.get('gt_boxes_2d'), \
                            data_dict['calib']
        new_shape = config.TARGET_SHAPE  # (h, w)
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
            1]  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = st.resize(img, new_unpad) * 255.0
        else:
            img = img.astype(np.float)
        top, bottom = int(dh // 2), int(dh - dh // 2)
        left, right = int(dw // 2), int(dw - dw // 2)
        img = np.pad(img, [(top, bottom), (left, right), (0, 0)])
        affine_matrix = np.asarray(
            [[r, 0.0, dw // 2],
             [0.0, r, dh // 2],
             [0, 0, 1]]
        )
        calib.P2 = affine_matrix @ calib.P2
        if self.training and boxes is not None:

            boxes *= r
            boxes[:, (0, 2)] += dw // 2
            boxes[:, (1, 3)] += dh // 2

        data_dict['images'] = img
        data_dict['gt_boxes_2d'] = boxes
        data_dict['calib'] = calib
        data_dict['image_valid_range'] = [left, top, left + new_unpad[1], top + new_unpad[0]]


        return data_dict

    def normalize_image(self, data_dict=None, config=None):
        if data_dict is None:
            new_config = EasyDict({
                'MEAN':np.asarray(config.MEAN)[:, None, None],
                'STD':np.asarray(config.STD)[:, None, None]
            })
            return partial(self.normalize_image, config=new_config)

        images = data_dict['images'].astype(np.float)
        images = np.ascontiguousarray(images.transpose(2, 0, 1))
        images = (images - config.MEAN) / config.STD
        data_dict['images'] = images
        return data_dict