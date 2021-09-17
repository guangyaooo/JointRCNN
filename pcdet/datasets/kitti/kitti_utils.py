import numpy as np
from ...utils import box_utils
from skimage import io


def pad_image(image, target_shape):
    '''
    Args:
        image: (H, W, 3)
        target_shape: (H', W')
        calib: Calibration
    Returns:
        image: (H, W, 3)
    '''
    old_h, old_w = image.shape[:2]
    new_h, new_w = target_shape
    image = np.pad(image,pad_width=[(0, new_h - old_h),
                                    (0, new_w - old_w),
                                    (0, 0)])
    return image

# def shift_boxes(boxes, org_shape, target_shape):
#     '''
#
#     Args:
#         boxes: (N, 4), [[xmin, ymin, xmax, ymax],...]
#         org_shape: tuple(H, W)
#         target_shape: tuple(H, W)
#
#     Returns:
#         boxes: (N, 4)
#     '''
#     old_h, old_w = org_shape
#     new_h, new_w = target_shape
#     pad_h = (new_h - old_h) // 2
#     pad_w = (new_w - old_w) // 2
#     boxes[:, 0] += pad_w
#     boxes[:, 2] += pad_w
#     boxes[:, 1] += pad_h
#     boxes[:, 3] += pad_h
#     return boxes


def get_fov_flag(pts_rect, img_shape, calib, return_pts_img = False):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    if return_pts_img:
        return pts_valid_flag, pts_img
    return pts_valid_flag

def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos
