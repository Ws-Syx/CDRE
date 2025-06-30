# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    # pre-defined by mask2former
    "ytvis_2019_train": ("/opt/data/private/syx/dataset/ytvis2019/train/JPEGImages",
                         "/opt/data/private/syx/dataset/ytvis2019/train.json"),
    "ytvis_2019_val": ("/opt/data/private/syx/dataset/ytvis2019/valid/JPEGImages",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT.json"),
    "ytvis_2019_test": ("ytvis2019/test/JPEGImages",
                        "ytvis2019/test.json"),
    
    # defined by myself
    # (1) training dataset
    "ytvis_2019_train_merge_v6_pair": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v6-pair-png.json"),

    # (2) evaluation dataset
    "ytvis_2019_val_x265_26": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-26",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_x265_29": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-29",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_x265_32": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-32",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_x265_35": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-35",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    
    "ytvis_2019_val_x264_26": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-26",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_x264_29": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-29",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_x264_32": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-32",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_x264_35": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-35",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    
    "ytvis_2019_val_dcvc_psnr_0": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-0",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_dcvc_psnr_1": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-1",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_dcvc_psnr_2": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-2",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_dcvc_psnr_3": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-3",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),

    "ytvis_2019_val_dcvc_dc_psnr_0": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-0",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_1": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-1",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_2": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-2",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_3": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-3",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
