import numpy as np
from skimage import io
import uuid
import cv2
import os.path as osp
import os
import pcdet.utils.box_utils as box_utils
from pcdet.structures import ImageList

def save_image(image, name = 'test.jpg'):
    io.imsave(f'../debug/{name}', image)

def save_image_with_boxes(img, bboxes=None, img_name=None, debug='../debug', save = True, color=None):
    color = color or (0, 255, 0)
    if img_name is None:
        img_name = str(uuid.uuid1()) + '.png'

    edges = [[0,1],
             [1,2],
             [2,3],
             [3,0],
             [0,4],
             [1,5],
             [5,4],
             [2,6],
             [5,6],
             [7,6],
             [7,3],
             [7,4]]
    if bboxes is not None:
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            if len(bbox.shape) == 1:
                # 2d box
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 1)
            else:
                # 3d box
                for start, end in edges:
                    p1 = bbox[start]
                    p2 = bbox[end]
                    cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color, 1)

    img = img.astype(np.uint8)
    if save:
        io.imsave(osp.join(debug, img_name), img)
    return img


def save_image_boxes_and_pts(img, bboxes, pts, calib, *, img_name=None, debug='../debug'):
    img = np.ascontiguousarray(img)
    img_pts = calib.lidar_to_img(pts)[0].astype(np.int)
    img = img.copy()
    img_pts[:, 0] = np.clip(img_pts[:, 0], 0, img.shape[1] - 1)
    img_pts[:, 1] = np.clip(img_pts[:, 1], 0, img.shape[0] - 1)
    img[img_pts[:, 1], img_pts[:, 0], :] = (255,0,0)
    img = img.astype(np.uint8)
    save_image_with_boxes(img, bboxes, img_name, debug)

def save_image_boxes_and_pts_labels_and_mask(img, bboxes, pts, pts_label, calib, pred_masks2d=[], *, img_name=None, debug='../debug'):
    img = np.ascontiguousarray(img)
    img_pts = calib.lidar_to_img(pts)[0].astype(np.int)
    img = img.copy()
    img_pts[:, 0] = np.clip(img_pts[:, 0], 0, img.shape[1] - 1)
    img_pts[:, 1] = np.clip(img_pts[:, 1], 0, img.shape[0] - 1)
    img[img_pts[:, 1], img_pts[:, 0], :] = (255,0,0)

    img_pts[:, 0] = np.clip(img_pts[:, 0], 0, img.shape[1] - 1)
    img_pts[:, 1] = np.clip(img_pts[:, 1], 0, img.shape[0] - 1)
    img[img_pts[:, 1], img_pts[:, 0], :] = (255, 0, 0)
    max_cls = pts_label.max() + 1
    colors = [
        (0, 255, 127),
        (0, 191, 255),
        (255, 255, 0),
        (255, 127, 80),
        (205, 92, 92)

    ]
    for j in range(1, max_cls):
        cls_ids = pts_label == j
        img[img_pts[cls_ids, 1], img_pts[cls_ids, 0], :] = \
        colors[j % len(colors)]

    if len(pred_masks2d) > 0:
        pred_masks2d = np.max(pred_masks2d, axis=0) > 0
        img[pred_masks2d] = img[pred_masks2d] * 0.5 + 122
        img = np.clip(img, 0, 255)

    img = img.astype(np.uint8)
    save_image_with_boxes(img, bboxes, img_name, debug)

def save_image_with_instances(data_batch=None, std = [1.0, 1.0, 1.0], mean=[103.53, 116.28, 123.675]):
    images = data_batch['images']
    if isinstance(images, ImageList):
        images = images.tensor
    instances = data_batch['instances']
    frame_ids = data_batch['frame_id']
    proposals = data_batch.get('proposals',None)

    std = np.asarray(std)[None, None, :]
    mean = np.asarray(mean)[None, None, :]

    pts_img = data_batch['pts_img']
    pts_batch_ids = data_batch['point_coords'][:, 0]
    pts_label = data_batch.get('pts_fake_target')
    colors = [
        (0,255,127),
        (0,191,255),
        (255,255,0),
        (255,127,80),
        (205,92,92)

    ]

    for i, (image, inst) in enumerate(zip(images, instances)):

        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.ascontiguousarray(image)
        image = std * image + mean
        bbox = inst.get('gt_boxes').tensor.cpu().numpy()
        fid = (frame_ids[i] + '.png') if frame_ids is not None else None
        image = save_image_with_boxes(img=image,
                                      bboxes=bbox,
                                      img_name=fid,
                                      save=(data_batch is None))
        if 'pred_mask2d' in data_batch:
            pred_masks2d = data_batch['pred_mask2d'][i].cpu().numpy()
            if len(pred_masks2d) > 0:
                pred_masks2d = np.max(pred_masks2d, axis=0) > 0
                image[pred_masks2d] = image[pred_masks2d] * 0.5 + 122
                image = np.clip(image, 0, 255)

        if pts_label is not None:
            single_pts_label = pts_label[pts_batch_ids == i].cpu().numpy()
            single_pts_img = pts_img[pts_batch_ids == i].cpu().numpy().astype(np.int)
            single_pts_img = single_pts_img[:,1:]
            single_pts_img[:, 0] = np.clip(single_pts_img[:, 0], 0, image.shape[1] - 1)
            single_pts_img[:, 1] = np.clip(single_pts_img[:, 1], 0, image.shape[0] - 1)
            image[single_pts_img[:, 1], single_pts_img[:, 0], :] = (255, 0, 0)
            max_cls = single_pts_label.max() + 1
            for j in range(1, max_cls):
                cls_ids = single_pts_label == j
                image[single_pts_img[cls_ids, 1], single_pts_img[cls_ids, 0], :] = colors[j]

            image = image.astype(np.uint8)


        if proposals is not None:
            bbox = proposals[i].get('proposal_boxes').tensor.cpu().numpy()
            logits = proposals[i].get('objectness_logits').cpu().numpy()
            top50_indices = np.argsort(logits)[:50]
            top50_bbox = bbox[top50_indices]
            image = save_image_with_boxes(img=image,
                                          bboxes=top50_bbox,
                                          img_name=fid,
                                          save=(data_batch is None),
                                          color=(0,0,255))



        if data_batch is not None:
            calib = data_batch['calib'][i]
            boxes3d = data_batch['gt_boxes'][i] if 'gt_boxes' in data_batch else []
            j = len(boxes3d) - 1
            while j>=0 and boxes3d[j].sum()==0:
                j -= 1
            if j >= 0:
                boxes3d = boxes3d[:j+1]
                boxes2d,boxes2d_corners = box_utils.lidar_box_to_image_box(boxes3d, calib)
                save_image_with_boxes(img=image,
                                      bboxes=boxes2d_corners,
                                      img_name=fid,
                                      save=True,
                                      color=(255,0,0))
            else:
                save_image_with_boxes(img=image,
                                      bboxes=None,
                                      img_name=fid,
                                      save=True,
                                      color=(255, 0, 0))





