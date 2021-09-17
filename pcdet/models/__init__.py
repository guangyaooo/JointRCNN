from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector
from ..structures import ImageList
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, _BatchNorm
from torch.nn.modules.instancenorm import InstanceNorm1d, InstanceNorm2d
def convert_bn2in(module):
    module_output = module
    if isinstance(module, BatchNorm2d):
        # instance_norm = InstanceNorm1d if isinstance(module, BatchNorm1d) else InstanceNorm2d
        module_output = InstanceNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_bn2in(child)
        )
    del module
    return module_output

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    if model_cfg.get('REPLACE_BN_WITH_IN', False):
        return convert_bn2in(model)
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key =='instances':
            batch_dict[key] = [x.to(torch.cuda.current_device()) for x in val]
            continue
        elif key == 'images':
            v = torch.from_numpy(val).float().cuda()
            batch_dict[key] = ImageList(v, [x.shape[-2:] for x in v])
            continue
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'calib_org', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'pred_dicts'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict, {})

    return model_func


def model_st_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'pred_dicts'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        pred_dicts = {}
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        for i, frame_id in enumerate(batch_dict['frame_id']):
            pred_dicts[frame_id] = {k:v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in ret_dict['pred_dicts'][i].items()}


        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict, pred_dicts)

    return model_func
