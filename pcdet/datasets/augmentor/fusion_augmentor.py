from functools import partial

import numpy as np
from skimage import transform as st
import cv2
from . import augmentor_utils
from ...utils import common_utils, debug_utils, box_utils
from .data_augmentor import DataAugmentor
from PIL import ImageEnhance, Image

class FusionAugmentor(DataAugmentor):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        super(FusionAugmentor, self).__init__(root_path, augmentor_configs, class_names, logger)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points, gt_boxes_2d, images, calib = \
            data_dict['gt_boxes'], data_dict['points'], data_dict[
                'gt_boxes_2d'], \
            data_dict['images'], data_dict['calib']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x']
            gt_boxes, points, gt_boxes_2d, images, calib = \
                augmentor_utils.random_flip_points_and_image(gt_boxes, points,
                                                             gt_boxes_2d,
                                                             images, calib)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['images'] = images
        data_dict['gt_boxes_2d'] = gt_boxes_2d
        data_dict['calib'] = calib
        return data_dict



    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, calib = augmentor_utils.global_fusion_rotation(
            data_dict['gt_boxes'], data_dict['points'], data_dict['calib'],
            rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['calib'] = calib

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, calib = augmentor_utils.global_fusion_scaling(
            data_dict['gt_boxes'], data_dict['points'], data_dict['calib'],
            config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['calib'] = calib
        return data_dict

    def color_jittering(self, data_dict=None, config=None):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        if data_dict is None:
            return partial(self.color_jittering, config=config)

        img = data_dict['images']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if config.brightness > 0:
            factor = np.random.uniform(-1, 1, 1) * config.brightness
            ImageEnhance.Brightness(img).enhance(factor)

        if config.contrast > 0:
            factor = np.random.uniform(-1, 1, 1) * config.contrast
            ImageEnhance.Contrast(img).enhance(factor)

        if config.sharpness > 0:
            factor = np.random.uniform(-1, 1, 1) * config.sharpness
            ImageEnhance.Sharpness(img).enhance(factor)

        if config.color > 0:
            factor = np.random.uniform(-1, 1, 1) * config.color
            ImageEnhance.Color(img).enhance(factor)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        data_dict['images'] = img

        return data_dict


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_boxes_2d'] = data_dict['gt_boxes_2d'][gt_boxes_mask]
            if 'gt_names' in data_dict:
                data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
