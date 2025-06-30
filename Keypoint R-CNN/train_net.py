#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import copy
import torch
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances


def add_custom_cfg(cfg):
    cfg.SOLVER.HEAD_MULTIPLIER = 0.1
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.BACKBONE_DISTORTION_EXTRACTOR_MULTIPLIER = 0.1
    cfg.SOLVER.BACKBONE_DISTORTION_EMBED_MULTIPLIER = 0.1
    cfg.SOLVER.BACKBONE_DISTORTION_LAYER_MULTIPLIER = 0.1


def dataset_register():
    # all keypoint dataset is coco2017-keypoint
    meta = _get_builtin_metadata("coco_person")
    register_coco_instances("keypoint_train", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_train2017.json", "/opt/data/private/syx/dataset/coco2017/train2017")
    register_coco_instances("keypoint_train_merge", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_train2017_merge_png.json", "/opt/data/private/syx/dataset/coco2017")
    register_coco_instances("keypoint_train_merge_pair", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_train2017_merge_pair_png.json", "/opt/data/private/syx/dataset/coco2017")
    
    register_coco_instances("keypoint_valid", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017.json", "/opt/data/private/syx/dataset/coco2017/val2017")
    register_coco_instances("keypoint_valid_pair", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_pair.json", "/opt/data/private/syx/dataset/coco2017/val2017")
    for i in [30, 35, 40, 45]:
        register_coco_instances(f"keypoint_valid_x264_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-x264-{i}")
        register_coco_instances(f"keypoint_valid_pair_x264_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_pair_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-x264-{i}")
    for i in [30, 35, 40, 45]:
        register_coco_instances(f"keypoint_valid_x265_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-x265-{i}")
        register_coco_instances(f"keypoint_valid_pair_x265_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_pair_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-x265-{i}")
    for i in [10, 20, 30, 40]:
        register_coco_instances(f"keypoint_valid_jpeg2000_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-jpeg2000-{i}")
        register_coco_instances(f"keypoint_valid_pair_jpeg2000_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_pair_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-jpeg2000-{i}")
    for i in [1, 2, 3, 4]:
        register_coco_instances(f"keypoint_valid_cheng2020_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-cheng2020-{i}")
        register_coco_instances(f"keypoint_valid_pair_cheng2020_{i}", meta, "/opt/data/private/syx/dataset/coco2017/annotations/person_keypoints_val2017_pair_png.json", f"/opt/data/private/syx/dataset/coco2017/val2017-png-cheng2020-{i}")

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        # weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    if "distortion_layers" in module_name:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_DISTORTION_LAYER_MULTIPLIER
                    elif "embed_layers" in module_name:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_DISTORTION_EMBED_MULTIPLIER
                    elif "distortion_extractor" in module_name:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_DISTORTION_EXTRACTOR_MULTIPLIER
                    else:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                else:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                                
                params.append({"params": [value], **hyperparams})
                print(f"Add into parameter with lr={hyperparams['lr']}: {module_name}")
             
        optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)

        return optimizer   

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file("config.yaml")
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    dataset_register()

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
