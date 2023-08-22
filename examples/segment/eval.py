import argparse

import yaml
import mindspore as ms
from addict import Dict
from mindspore.communication import get_group_size, get_rank, init
from mindspore import log as logger

from data import create_segment_dataset
from deeplabv3 import DeepLabV3, DeepLabV3InferNetwork
from deeplab_resnet import *
from postprocess import apply_eval

from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import (
    AllReduceSum,
    StateMonitor,
    create_trainer,
    get_metrics,
    require_customized_train_step,
    set_logger,
    set_seed,
)
from mindspore import load_checkpoint, load_param_into_net

def check_batch_size(num_samples, ori_batch_size=32, refine=True):
    if num_samples % ori_batch_size == 0:
        return ori_batch_size
    else:
        # search a batch size that is divisible by num samples.
        for bs in range(ori_batch_size - 1, 0, -1):
            if num_samples % bs == 0:
                print(
                    f"WARNING: num eval samples {num_samples} can not be divided by "
                    f"the input batch size {ori_batch_size}. The batch size is refined to {bs}"
                )
                return bs
    return 1


def eval(args):
    # create dataset and load
    eval_dataset = create_segment_dataset(
        name=args.dataset,
        image_mean=args.image_mean,
        image_std=args.image_std,
        data_dir=args.data_dir,
        crop_size=args.crop_size,
        num_classes=args.num_classes,
        num_parallel_workers=args.num_parallel_workers,
        shuffle=False,
        is_training=False, 
    )
    
    # check batch size
    args.batch_size = check_batch_size(eval_dataset.get_dataset_size(), args.batch_size)
    
    # create eval model
    backbone = create_model(
        args.backbone,
        features_only=args.backbone_features_only,
        out_indices=args.backbone_out_indices,
        output_stride=args.output_stride,
    )
    deeplabv3 = DeepLabV3(backbone, args, is_training=False)
    eval_model = DeepLabV3InferNetwork(deeplabv3, input_format=args.input_format)
    eval_model.init_parameters_data()

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_model, param_dict)
    eval_model.set_train(False)   

    print("\n========================================\n")
    print("Processing, please wait a moment.")

    # evaluate
    eval_param_dict = {"net": eval_model, 
                       "dataset": eval_dataset, 
                       "args": args}
    
    Iou, mIoU = apply_eval(eval_param_dict)
      
    print("\n========================================\n")
    print(f"per-class IoU: {Iou}")
    print(f"mean IoU: {mIoU}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Config", add_help=False)
    parser.add_argument(
        "-c", "--config", type=str, default="", help="YAML config file specifying default arguments (default=" ")"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    yaml_fp = args.config

    with open(yaml_fp) as fp:
        args = yaml.safe_load(fp)

    args = Dict(args)

    # core evaluation
    eval(args)
   
