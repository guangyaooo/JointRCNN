import numpy as np

from ...utils import common_utils
from pcdet.utils import debug_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_points_and_image(gt_boxes, points,
                                 gt_boxes2d, image, calib):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        image: (H, W, 3)
        calib: Calibration
    Returns:
    """
    # debug_utils.save_image_boxes_and_pts(image, gt_boxes2d, points[:,:3], calib,
    #                                      img_name='org.jpg')

    pts_enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if pts_enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

        pts_reverse = np.identity(4, dtype=calib.V2C.dtype)
        pts_reverse[1, 1] = -1
        calib.V2C = calib.V2C @ pts_reverse

    img_enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if img_enable:
        h, w, _ = image.shape
        image = np.ascontiguousarray(image[:, ::-1, :])

        x_min, y_min, x_max, y_max = np.split(gt_boxes2d, 4, axis=1)
        gt_boxes2d = np.concatenate(
            [w - x_max, y_min, w - x_min, y_max], axis=1
        ).astype(gt_boxes2d.dtype)

        P = np.asarray([
            [-1.0, 0.0, w],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        calib.P2 = P @ calib.P2

    # debug_utils.save_image_boxes_and_pts(image, gt_boxes2d, points[:,:3], calib,
    #                                      img_name='aug.jpg')

    return gt_boxes, points, gt_boxes2d, image, calib


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = \
    common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                       np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def global_fusion_rotation(gt_boxes, points, calib, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max],
        calib: Calibration
    Returns:
    """
    # debug_utils.save_image_boxes_and_pts(image, None, points[:,:3], calib,
    #                                      img_name='org.jpg')
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])

    cosa = np.cos(-noise_rotation)
    sina = np.sin(-noise_rotation)
    inverse_rot_matrix = np.asarray([
        [cosa, -sina, 0.0, 0.0],
        [sina, cosa, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=calib.V2C.dtype)
    calib.V2C = calib.V2C @ inverse_rot_matrix

    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = \
    common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                       np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    # debug_utils.save_image_boxes_and_pts(image, None, points[:, :3], calib,
    #                                      img_name='aug.jpg')

    return gt_boxes, points, calib


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_fusion_scaling(gt_boxes, points, calib, scale_range):
    """
    Args:
        calib: Calibration
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    # debug_utils.save_image_boxes_and_pts(image, None, points[:,:3], calib,
    #                                      img_name='org.jpg')
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    inverse_scale = 1.0/noise_scale
    inverse_scale_matrix = np.asarray([
        [inverse_scale, 0.0, 0.0, 0.0],
        [0, inverse_scale, 0.0, 0.0],
        [0.0, 0.0, inverse_scale, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=calib.V2C.dtype)
    calib.V2C = calib.V2C @ inverse_scale_matrix

    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    # debug_utils.save_image_boxes_and_pts(image, None, points[:, :3], calib,
    #                                      img_name='aug.jpg')
    return gt_boxes, points, calib
