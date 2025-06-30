# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

if __name__ == "__main__":
        
    # create waymo dataset
    try:
        train_image_path = "/opt/data/private/syx/dataset/ytvis2019/train/JPEGImages"
        train_json_path = "/opt/data/private/syx/dataset/ytvis2019-coco/train-coco.json"

        valid_image_path = "/opt/data/private/syx/dataset/ytvis2019/valid/JPEGImages"
        valid_json_path = "/opt/data/private/syx/dataset/ytvis2019-coco/valid-coco.json"
        register_coco_instances("ytvis_cocoformat_train", {}, train_json_path, train_image_path)
        register_coco_instances("ytvis_cocoformat_valid", {}, valid_json_path, valid_image_path)
    except:
        print("Error at register_coco_instances.")
    
    # prepare the config information
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.merge_from_file("config.yaml")
    cfg.freeze()
    print(cfg)
    
    # training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # validation
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    # # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    # predictor = DefaultPredictor(cfg)
    # evaluator = COCOEvaluator("waymo_valid", output_dir=cfg.OUTPUT_DIR)
    # val_loader = build_detection_test_loader(cfg, "waymo_valid")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`