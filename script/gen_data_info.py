from pcdet.datasets.kitti.kitti_fusion_dataset import create_kitti_infos
import sys

import yaml
from pathlib import Path
from easydict import EasyDict
dataset_cfg = EasyDict(yaml.load(open(sys.argv[1])))
ROOT_DIR = (Path(__file__).resolve().parent / '../' ).resolve()
create_kitti_infos(
    dataset_cfg=dataset_cfg,
    class_names=['Car'],
    data_path=Path(dataset_cfg.DATA_PATH),
    save_path=Path(dataset_cfg.DATA_PATH)
)