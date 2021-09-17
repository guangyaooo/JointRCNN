import torch

from pcdet.structures import Instances, Boxes
from pcdet.utils import box_utils
from .pointrcnn_head import PointRCNNHead
from ..d2_proposal_generator import build_proposal_generator
from ..d2_roi_heads import build_roi_heads
from pcdet.utils import debug_utils
from pcdet.utils import loss_utils
import torch.nn.functional as F

class FusionRCNNHead(PointRCNNHead):

    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, num_class)
        model_cfg.IMAGE_HEAD_NUM_CLASSES = kwargs.get('image_head_num_classes')
        if model_cfg.TRAIN_IMAGE_BOXHEAD:
            model_cfg.IMAGE_HEAD_NUM_CLASSES = kwargs.get('real_classes')
        # BG_LABEL was used in image roi head to label the background class
        model_cfg.BG_LABEL = 0
        self.real_num_class = kwargs.get('real_classes')

        self.image_rpn = build_proposal_generator(model_cfg,
                                                  kwargs.get('input_shape'))
        self.image_roi_head = build_roi_heads(model_cfg,
                                              kwargs.get('input_shape'))
        self.size_divisibility = kwargs.get('size_divisibility')
        self.add_module(
            'da_box_cls_loss',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    # TODO combine point rcnn and faster rcnn
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                ------------------------------------> used in proposal layer

                gt_boxes: (B, N, 7 + C + 1) ->
                ------------------------------------> use in proposal target layer

                point_features: (num_points, C)
                ------------------------------------> used in roi_pool3d
            nms_config:

            batch_size:
        rois: (B, num_rois, 7 + C)
        point_coords: (num_points, 4)  [bs_idx, x, y, z]
        point_features: (num_points, C)
        point_cls_scores: (N1 + N2 + N3 + ..., 1)
        point_part_offset: (N1 + N2 + N3 + ..., 3)


        Returns:

        """
        self_training_mode = batch_dict['self_training_mode']
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG[
                'TRAIN' if self.training else 'TEST']
        )
        image_losses = {}
        gt_instances = batch_dict['instances'] if self.training else None
        images = batch_dict['images']
        features = batch_dict['top_down_features']

        image_info_dict = {}
        rois_keep_2d = []
        if self.image_rpn is None:
            roi3d = batch_dict['rois'].detach()
            roi3d_scores = batch_dict['roi_scores'].detach()
            proposals = []
            for i in range(batch_dict['batch_size']):
                calib = batch_dict['calib'][i]
                image_valid_range = batch_dict['image_valid_range'][i]
                cur_roi2d, _ = box_utils.lidar_box_to_image_box(roi3d[i], calib)

                cur_roi2d[:, 0].clamp_(image_valid_range[0], image_valid_range[2] - 1)
                cur_roi2d[:, 2].clamp_(image_valid_range[0], image_valid_range[2] - 1)
                cur_roi2d[:, 1].clamp_(image_valid_range[1], image_valid_range[3] - 1)
                cur_roi2d[:, 3].clamp_(image_valid_range[1], image_valid_range[3] - 1)
                inst = Instances(images[i].shape[-2:])
                boxes = Boxes(cur_roi2d)
                keep = boxes.nonempty()
                rois_keep_2d.append(keep)
                logits = roi3d_scores[i][keep]
                boxes = boxes[keep]
                inst.set('proposal_boxes', boxes)
                inst.set('objectness_logits', logits)
                proposals.append(inst)
            # tod remove debug code
            # batch_dict['proposals'] = proposals
            # debug_utils.save_image_with_instances(data_batch=batch_dict)
        else:
            # if '014984' == batch_dict['frame_id'][-1]:
            #     print()
            proposals, image_proposal_losses = self.image_rpn(images,
                                                              features,
                                                              gt_instances,
                                                              self_training_mode)

            image_losses.update(image_proposal_losses)
        if self.image_roi_head.training:
            proposals = self.image_roi_head.label_and_sample_proposals(
                proposals, gt_instances)

            for cls_id in range(self.real_num_class + 1):
                image_info_dict['sampled_proposals/class %d num' % cls_id] = \
                    (proposals[0].get('gt_classes') == cls_id).sum().item()

        pred_instances, \
        image_roi_losses, \
        image_pooled_features, \
        boxes2d, scores2d, \
        box_keep2d \
            = self.image_roi_head( images, features, proposals, gt_instances, self_training_mode)
        image_losses.update(image_roi_losses)
        use_gt_label = 'gt_boxes' in batch_dict
        if self.training:

            if use_gt_label:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
            else:
                # TODO sample ROIs
                targets_dict = self.sample_rois_with_box2d(rois_keep_2d, [x[:, 3].detach() for x in scores2d], batch_dict)

                batch_dict['rois'] = targets_dict['rois']
                batch_dict['soft_box_labels'] = targets_dict['soft_box_labels']

        pooled_features = self.roipool3d_gpu(
            batch_dict)  # (total_rois, num_sampled_points, 3 + C)

        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(
            1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[...,
                         self.num_prefix_channels:].transpose(1, 2).unsqueeze(
            dim=3)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)

        # TODO: 在这里获取逐点对应的image特征，进行融合

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [
            merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, points_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]  # (total_rois, num_features, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'],
                cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['batch_cls_preds2d'] = scores2d
            batch_dict['batch_box_preds2d'] = boxes2d
            batch_dict['batch_roi_keep2d'] = rois_keep_2d
            batch_dict['batch_box_keep2d'] = box_keep2d

            batch_dict['image_preds'] = pred_instances
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            if not self.image_roi_head.training:
                batch_dict['image_preds'] = pred_instances
            if self_training_mode:
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'],
                    rois=batch_dict['rois'],
                    cls_preds=rcnn_cls, box_preds=rcnn_reg
                )
                batch_dict['batch_cls_preds'] = batch_cls_preds
                batch_dict['batch_box_preds'] = batch_box_preds
                batch_dict['batch_cls_preds2d'] = scores2d
                batch_dict['batch_box_preds2d'] = boxes2d
                batch_dict['batch_roi_keep2d'] = rois_keep_2d
                batch_dict['batch_box_keep2d'] = box_keep2d

                batch_dict['image_preds'] = pred_instances
                batch_dict['cls_preds_normalized'] = False


            self.forward_ret_dict = targets_dict
            self.forward_ret_dict['image_losses'] = image_losses
            self.forward_ret_dict['image_info'] = (image_info_dict)
            self.forward_ret_dict['ignore_box_loss'] = batch_dict.get('ignore_box_loss', False)
        return batch_dict

    def sample_rois_with_box2d(self, rois_keep_2d, scores2d, batch_dict):

        cfg = self.model_cfg.TARGET_CONFIG

        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']

        fg_nums = int(cfg.ROI_PER_IMAGE * cfg.FG_RATIO)
        bg_nums = int(cfg.ROI_PER_IMAGE - fg_nums)

        # TODO support multiple classes
        sampled_rois = []
        soft_boxes_label = []
        for index in range(batch_size):
            single_sampled_rois = []
            single_box_labels = []
            single_roi = rois[index]
            single_score2d = scores2d[index]
            single_roi_keep = torch.nonzero(rois_keep_2d[index], as_tuple=True)[0]
            sort_index = torch.argsort(single_score2d)
            # fg
            single_sampled_rois.append(single_roi[single_roi_keep[sort_index[-fg_nums:]]])
            single_box_labels.append(single_score2d[sort_index[-fg_nums:]])

            # bg
            single_sampled_rois.append(single_roi[single_roi_keep[sort_index[:bg_nums]]])
            single_box_labels.append(single_score2d[sort_index[:bg_nums]])

            single_sampled_rois = torch.cat(single_sampled_rois, dim=0).unsqueeze(0)
            single_box_labels = torch.cat(single_box_labels, dim=0)

            sampled_rois.append(single_sampled_rois)
            soft_boxes_label.append(single_box_labels)

        sampled_rois = torch.cat(sampled_rois, dim=0)
        soft_boxes_label = torch.cat(soft_boxes_label, dim=0)
        target_dict = {
            'rois': sampled_rois,
            'soft_box_labels': soft_boxes_label
        }
        return target_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        if 'rcnn_cls_labels' in forward_ret_dict:
            rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

            if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
                rcnn_cls_flat = rcnn_cls.view(-1)
                batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
                cls_valid_mask = (rcnn_cls_labels >= 0).float()
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
                batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
                cls_valid_mask = (rcnn_cls_labels >= 0).float()
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            else:
                raise NotImplementedError
        else:
            soft_box_labels = forward_ret_dict['soft_box_labels']
            rcnn_loss_cls = self.da_box_cls_loss(rcnn_cls.view(-1), soft_box_labels, torch.ones_like(soft_box_labels)).mean()

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        # image losses
        image_loss_dict = self.forward_ret_dict['image_losses']
        image_losses = 0
        if len(image_loss_dict) > 0:
            image_losses = sum(image_loss_dict.values())
            tb_dict['image_total_loss'] = image_losses.item()
            tb_dict.update(
                {'image_' + k: v.item() for k, v in image_loss_dict.items()}
            )

        tb_dict.update(self.forward_ret_dict.get('image_info', {}))

        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(
            self.forward_ret_dict)

        tb_dict.update(cls_tb_dict)
        if 'gt_of_rois' in self.forward_ret_dict and not self.forward_ret_dict.get('ignore_box_loss', False):
            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(
                self.forward_ret_dict)
        else:
            rcnn_loss_reg, reg_tb_dict = 0, {}

        rcnn_loss = rcnn_loss_reg + rcnn_loss_cls + image_losses
        # rcnn_loss = rcnn_loss_reg + rcnn_loss_cls

        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict
