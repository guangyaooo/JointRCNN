import pickle
from collections import OrderedDict
from typing import Dict, Any

import numpy as np
import torch
from fvcore.common.checkpoint import _strip_prefix_if_present
from fvcore.common.file_io import PathManager

from pcdet.layers.mask_ops import paste_masks_in_image
from pcdet.models.d2_backbone import FPN
from pcdet.models.d2_proposal_generator import RPN
from pcdet.models.d2_roi_heads import StandardROIHeads
from pcdet.structures import Boxes, Instances
from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils
from ...utils.box_utils import recover_boxes_2d, boxes_iou_normal, \
    lidar_box_to_image_box, in_2d_box
from ...utils.memory import retry_if_cuda_oom
from pcdet.utils import debug_utils

from torch.nn.functional import grid_sample

class FusionRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        if model_cfg.FREEZE_IMAGE_BRANCH:
            model_cfg.ROI_HEAD.TRAIN_IMAGE_BOXHEAD = False

        super().__init__(model_cfg=model_cfg, num_class=num_class,
                         dataset=dataset)
        self.module_list = self.build_networks()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        # set image branch mode to eval
        if self.model_cfg.FREEZE_IMAGE_BRANCH:
            self.module_list[0].eval()
            if self.module_list[3].image_rpn is not None:
                self.module_list[3].image_rpn.eval()
            self.module_list[3].image_roi_head.eval()

        return self
    def forward(self, batch_dict):
        # # TODO debug
        # debug_utils.save_image_with_instances(data_batch=batch_dict,)
        batch_dict['self_training_mode'] = self.model_cfg.get('SELF_TRAINING', False)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            if batch_dict['self_training_mode']:
                pred_dicts, recall_dicts = self.fusion_post_processing9(
                    batch_dict,ret_recall=False)
                ret_dict['pred_dicts'] = pred_dicts

            return ret_dict, tb_dict, disp_dict
        else:
            if self.model_cfg.POST_PROCESSING.get('FUSION', False) and \
                    (self.model_cfg.ROI_HEAD.IMAGE_PROPOSAL_GENERATOR.NAME == 'PrecomputedProposals' or
                     self.model_cfg.POST_PROCESSING.get('STRATEGY',0) == 9):
                fusion_strategy = self.model_cfg.POST_PROCESSING.get('STRATEGY',
                                                                     0)
                pred_dicts, recall_dicts = getattr(self,
                                                   'fusion_post_processing%d' % fusion_strategy)(
                    batch_dict)
            else:
                # pred_dicts, recall_dicts = self.post_processing(batch_dict)
                pred_dicts, recall_dicts = self.fusion_post_processing6(
                    batch_dict)

            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        tb_dict['point_total_loss'] = tb_dict.get('rcnn_loss_reg', 0) + \
                                      tb_dict['rcnn_loss_cls'] + \
                                      tb_dict['point_loss_cls'] + \
                                      tb_dict.get('point_loss_box', 0)

        loss = loss_point + loss_rcnn
        disp_dict['point_loss'] = tb_dict['point_total_loss']
        if 'image_total_loss' in tb_dict:
            disp_dict['image_loss'] = tb_dict['image_total_loss']

        return loss, tb_dict, disp_dict

    def split_parameters(self):
        '''Retrun point branch parameters and image branch parameters
        Returns:

        '''
        image_branch_types = (
            FPN,
            RPN,
            StandardROIHeads,
        )
        point_params = []
        image_params = []
        memo = set()
        for module in self.modules():
            if isinstance(module, image_branch_types):
                memo = memo.union(module.parameters())
        for module in self.modules():
            if isinstance(module, image_branch_types):
                continue
            for p in module.parameters(recurse=False):
                if p not in memo:
                    memo.add(p)
                    point_params.append(p)

        image_params.extend(self.module_list[0].parameters())
        image_params.extend(self.module_list[3].image_roi_head.parameters())
        if self.module_list[3].image_rpn is not None:
            image_params.extend(self.module_list[3].image_rpn.parameters())


        return point_params, image_params

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v,
                                                                torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k,
                                                                          type(
                                                                              v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

    def split_state_dict(self, state_dict: Dict[str, Any], prefix):
        gropu1 = OrderedDict()
        gropu2 = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(prefix):
                gropu1[key] = value
            else:
                gropu2[key] = value
        return gropu1, gropu2

    def load_params_for_img_branch(self, logger):
        image_branch_weight = self.model_cfg.get('IMAGE_BRANCH_WEIGHT', '')
        if image_branch_weight != '':

            if not self.model_cfg.ROI_HEAD.TRAIN_IMAGE_BOXHEAD:
                if self.model_cfg.IMAGE_PRETRAINED_DATASET == 'cityscapes':
                    self.image_thing_classes = ['person', 'rider', 'car',
                                                'truck',
                                                'bus', 'train', 'motorcycle',
                                                'bicycle']
                    cls_map = {
                        'Car': 3,
                        'Pedestrian': 1,
                        # 'Cyclist': [7, 8],
                    }
                    self.register_buffer('cls_map',
                                         torch.zeros(9, dtype=torch.long))
                    self.register_buffer('inverse_cls_map',
                                         torch.zeros(len(
                                             self.model_cfg.CLASS_NAMES) + 1,
                                                     dtype=torch.long))
                    for i, cls in enumerate(self.model_cfg.CLASS_NAMES):
                        assert cls in cls_map, \
                            'cityscape has no class mapped to %s' % cls
                        mapped_cls = cls_map[cls]
                        if isinstance(mapped_cls, list):
                            for k in mapped_cls:
                                self.cls_map[k] = i + 1
                                self.inverse_cls_map[i + 1] = k
                        else:
                            self.cls_map[mapped_cls] = i + 1
                            self.inverse_cls_map[i + 1] = mapped_cls

                else:
                    raise NotImplementedError

            logger.info(
                'Loading image branch weights from %s' % image_branch_weight)
            with PathManager.open(image_branch_weight, "rb") as f:
                image_branch_state_dict = pickle.load(f, encoding="latin1")

            image_branch_state_dict = image_branch_state_dict.pop('model')
            self._convert_ndarray_to_tensor(image_branch_state_dict)
            image_backbone, others = self.split_state_dict(
                image_branch_state_dict, prefix='backbone')
            _strip_prefix_if_present(image_backbone, 'backbone.')
            self.module_list[0].load_state_dict(image_backbone, True)

            proposal_generator, roi_heads = self.split_state_dict(others,
                                                                  prefix='proposal_generator')
            _strip_prefix_if_present(proposal_generator, 'proposal_generator.')
            _strip_prefix_if_present(roi_heads, 'roi_heads.')
            if self.module_list[3].image_rpn is not None:
                self.module_list[3].image_rpn.load_state_dict(
                    proposal_generator, True)
            if self.model_cfg.ROI_HEAD.TRAIN_IMAGE_BOXHEAD:
                _, roi_heads = self.split_state_dict(roi_heads,
                                                     prefix='box_predictor')
                logger.info('Train image box head, pop box_predictor weights')
            self.module_list[3].image_roi_head.load_state_dict(roi_heads, False)
            # swap the first channel
            with torch.no_grad():
                cls_score = self.module_list[
                    3].image_roi_head.box_predictor.cls_score

                # cls_score: Linear(in_features=1024,out_features=num_classes+1)
                weight_copy = cls_score.weight.clone()
                bias_copy = cls_score.bias.clone()
                out_channels = cls_score.weight.shape[0]

                cls_score.weight[1:out_channels, ...] = weight_copy[
                                                        :out_channels - 1]
                cls_score.weight[0, ...] = weight_copy[-1, ...]

                cls_score.bias[1:out_channels, ...] = bias_copy[
                                                      :out_channels - 1]
                cls_score.bias[0, ...] = bias_copy[-1, ...]

            logger.info('Successfully load image branch weight!')

    def load_params_from_file(self, filename, logger, to_cpu=False):
        super(FusionRCNN, self).load_params_from_file(filename, logger, to_cpu)
        if self.training:
            self.load_params_for_img_branch(logger)


    def fusion_post_processing6(self, batch_dict):
        # TODO Align 2d, 3d results
        pred_dicts, recall_dicts = super(FusionRCNN, self).post_processing(
            batch_dict)
        pred_instances = batch_dict['image_preds']
        # TODO post processing pred mask
        for image_shape, pred3d, pred2d in zip(batch_dict['image_shape'],
                                               pred_dicts, pred_instances):
            new_shape = np.array(pred2d.image_size)
            pred_boxes2d = pred2d.get('pred_boxes').tensor
            pred_labels2d = pred2d.get('pred_classes')
            pred_scores2d = pred2d.get('scores')
            pred_masks2d = pred2d.get('pred_masks') if pred2d.has('pred_masks') else torch.zeros((len(pred_boxes2d),1, 14, 14), device=pred_boxes2d.device, dtype=torch.bool)
            pred_boxes2d = recover_boxes_2d(pred_boxes2d,
                                            image_shape,
                                            new_shape)
            if hasattr(self, 'cls_map'):
                pred_labels2d = self.cls_map[pred_labels2d]
                mask = pred_labels2d > 0
                pred_labels2d = pred_labels2d[mask]
                pred_scores2d = pred_scores2d[mask]
                pred_boxes2d = pred_boxes2d[mask]
                pred_masks2d = pred_masks2d[mask]

            pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
                pred_masks2d[:, 0, :, :],  # N, 1, M, M
                Boxes(pred_boxes2d),
                tuple(image_shape),
                threshold=0.5,
            )

            pred3d.update({
                'pred_boxes2d': pred_boxes2d.cpu().numpy(),
                'pred_scores2d': pred_scores2d.cpu().numpy(),
                'pred_labels2d': pred_labels2d.cpu().numpy(),
                'pred_masks2d': pred_masks.cpu().numpy(),
                'pred_masks2d_org': pred_masks2d.cpu().numpy()
            })



        return pred_dicts, recall_dicts


    def fusion_post_processing9(self, batch_dict, ret_recall=True):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        updated_instances = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            org_shape = batch_dict['image_shape'][index]
            cur_shape = batch_dict['images'].image_sizes[index]
            image_valid_range = batch_dict['image_valid_range'][index]
            calib = batch_dict['calib'][index]
            # roi_keep_2d = batch_dict['batch_roi_keep2d'][index]

            # fetch and process 3d box preds, cls preds and label preds
            box_preds_3d = batch_dict['batch_box_preds'][batch_mask].detach()
            # box_preds_3d = box_preds_3d[roi_keep_2d]
            src_box_preds_3d = box_preds_3d

            cls_preds_3d = batch_dict['batch_cls_preds'][batch_mask].detach()
            # cls_preds_3d = cls_preds_3d[roi_keep_2d]
            src_cls_preds_3d = cls_preds_3d
            assert cls_preds_3d.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds_3d = torch.sigmoid(cls_preds_3d)

            cls_preds_3d, label_preds_3d = torch.max(cls_preds_3d, dim=-1)
            if batch_dict.get('has_class_labels', False):
                label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                label_preds_3d = batch_dict[label_key][index]
                # label_preds_3d = label_preds_3d[roi_keep_2d]
            else:
                label_preds_3d = label_preds_3d + 1

            select_from_3d, selected_scores_3d = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds_3d, box_preds=box_preds_3d,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            # fetch and process 2d box preds, cls preds and label preds
            select_from_2d = batch_dict['batch_box_keep2d'][index].detach()

            # if batch_dict['frame_id'][index] == '000008':
            #     print()
            preds_dict_2d = batch_dict['image_preds'][index]
            selected_boxes_2d = preds_dict_2d.get('pred_boxes').tensor.double().detach()
            label_preds_2d = preds_dict_2d.get('pred_classes').detach()
            cls_preds_2d = preds_dict_2d.get('scores').detach()

            if hasattr(self, 'cls_map'):
                label_preds_2d = self.cls_map[label_preds_2d]
                mask = label_preds_2d > 0
                select_from_2d = select_from_2d[mask]
                label_preds_2d = label_preds_2d[mask]
                cls_preds_2d = cls_preds_2d[mask]
                selected_boxes_2d = selected_boxes_2d[mask]

            # union 2d & 3d results
            final_scores_3d = []
            final_scores_2d = []
            final_labels_3d = []
            final_labels_2d = []
            final_boxes_2d = []
            final_boxes_3d = []
            only_in_3d = 0
            only_in_2d = 0
            both = 0
            only3d_ious = []
            only2d_ious = []
            both_ious = []

            select_project_box = box_preds_3d[select_from_3d]
            select_project_box, _ = lidar_box_to_image_box(select_project_box,
                                                           calib)
            select_project_box[:, 0].clamp_(image_valid_range[0],
                                            image_valid_range[2] - 1)
            select_project_box[:, 2].clamp_(image_valid_range[0],
                                            image_valid_range[2] - 1)
            select_project_box[:, 1].clamp_(image_valid_range[1],
                                            image_valid_range[3] - 1)
            select_project_box[:, 3].clamp_(image_valid_range[1],
                                            image_valid_range[3] - 1)
            iou_matrix = boxes_iou_normal(selected_boxes_2d, select_project_box)
            if iou_matrix.size(0) == 0 or iou_matrix.size(1) == 0:
                iou_matrix = torch.zeros((max(1, iou_matrix.size(0)),
                                          max(1, iou_matrix.size(1))),
                                         device=iou_matrix.device)

            indices_2d = set(select_from_2d.cpu().numpy())
            indices_3d = set(select_from_3d.cpu().numpy())

            iou_thresh = self.model_cfg.POST_PROCESSING.IOU_THRESH

            inverse_idx_2d = {x.item(): i for i, x in
                              enumerate(select_from_2d)}
            inverse_idx_3d = {x.item(): i for i, x in
                              enumerate(select_from_3d)}

            indices_2d = sorted(list(indices_2d))
            indices_3d = sorted(list(indices_3d))

            non_match = False
            while len(indices_2d) > 0 or len(indices_3d) > 0:
                if len(indices_2d) == 0 or len(
                        indices_3d) == 0 or non_match:
                    break
                x_idx = [inverse_idx_2d[ii] for ii in indices_2d]
                y_idx = [inverse_idx_3d[jj] for jj in indices_3d]
                sub_iou_matrix = iou_matrix[x_idx][:, y_idx]
                max_idx = torch.argmax(sub_iou_matrix)
                _j, _i = int(max_idx // sub_iou_matrix.size(1)), int(
                    max_idx % sub_iou_matrix.size(1))

                if sub_iou_matrix[_j, _i] >= iou_thresh:  # TODO
                    idx3d = indices_3d.pop(_i)
                    idx2d = indices_2d.pop(_j)
                    _j = inverse_idx_2d[idx2d]
                    _i = inverse_idx_3d[idx3d]
                    final_scores_2d.append(cls_preds_2d[_j])
                    final_labels_2d.append(label_preds_2d[_j])
                    final_boxes_2d.append(selected_boxes_2d[_j])
                    final_scores_3d.append(cls_preds_3d[idx3d])
                    final_labels_3d.append(label_preds_3d[idx3d])
                    final_boxes_3d.append(box_preds_3d[idx3d])
                    both_ious.append(iou_matrix[_j, _i].item())
                    both += 1
                else:
                    non_match = True

            device = cls_preds_3d.device
            final_scores_3d = torch.tensor(final_scores_3d, device=device)
            final_labels_3d = torch.tensor(final_labels_3d, device=device)
            final_boxes_3d = torch.cat([x.view(1, -1) for x in final_boxes_3d],
                                       dim=0) \
                if len(final_boxes_3d) > 0 else torch.tensor([],
                                                             device=device).view(
                0, 7)

            final_labels_2d = torch.tensor(final_labels_2d, device=device)
            final_boxes_2d = torch.cat(
                [x.view(1, -1) for x in final_boxes_2d], dim=0) \
                if len(final_boxes_2d) > 0 else torch.tensor([],
                                                             device=device).view(
                0, 4)
            final_scores_2d = torch.tensor(final_scores_2d, device=device)
            inverse_labels_2d = final_labels_2d.long()
            if hasattr(self,'inverse_cls_map') and len(inverse_labels_2d) > 0:
                inverse_labels_2d = self.inverse_cls_map[inverse_labels_2d]
            new_instance = Instances(preds_dict_2d.image_size,
                                     pred_boxes=Boxes(final_boxes_2d),
                                     pred_classes=inverse_labels_2d,
                                     scores=final_scores_2d)

            if len(final_boxes_2d) > 0:
                final_boxes_2d = recover_boxes_2d(final_boxes_2d,
                                                  org_shape,
                                                  cur_shape)
            new_instance.set('reshaped_boxes', Boxes(final_boxes_2d))
            updated_instances.append(new_instance)

            if ret_recall:
                recall_dict = self.generate_recall_record(
                    box_preds=final_boxes_3d if 'rois' not in batch_dict else src_box_preds_3d,
                    recall_dict=recall_dict, batch_index=index,
                    data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )

            record_dict = {
                'pred_boxes': final_boxes_3d,
                'pred_scores': final_scores_3d,
                'pred_labels': final_labels_3d,

                'pred_boxes2d': final_boxes_2d.cpu().numpy(),
                'pred_scores2d': final_scores_2d.cpu().numpy(),
                'pred_labels2d': final_labels_2d.cpu().numpy(),
                'only_in_2d': only_in_2d,
                'only_in_3d': only_in_3d,
                'both': both,
                'only3d_ious': only3d_ious,
                'only2d_ious': only2d_ious,
                'both_ious': both_ious
            }
            assert len(final_boxes_3d) >= len(final_boxes_2d)
            pred_dicts.append(record_dict)


        if hasattr(self,'cls_map'):
            self.module_list[3].image_roi_head.forward_with_given_boxes(
                batch_dict['top_down_features'],
                updated_instances
            )
            for i, instance in enumerate(updated_instances):

                pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
                    instance.pred_masks[:, 0, :, :],  # N, 1, M, M
                    instance.reshaped_boxes,
                    tuple(batch_dict['image_shape'][i]),
                    threshold=0.5,
                )
                pred_dicts[i]['pred_masks2d'] = pred_masks.cpu().numpy()
        else:
            for i in range(batch_size):
                pred_dicts[i]['pred_masks2d'] = []

        return pred_dicts, recall_dict

