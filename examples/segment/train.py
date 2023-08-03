import argparse
import os
import sys

import yaml
# from callbacks import get_ssd_callbacks, get_ssd_eval_callback
# from data import create_ssd_dataset
# from ssd_model import SSD, SSDInferWithDecoder, SSDWithLossCell, get_ssd_trainer
# from utils import get_ssd_lr_scheduler, get_ssd_optimizer

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init
from data import create_segment_dataset
from loss import SoftmaxCrossEntropyLoss
from deeplabv3 import DeepLabV3, DeepLabV3WithLoss

from mindcv.models import create_model
from mindcv.utils import set_seed

def train(args):
    """main train function"""
    ms.set_context(mode=args.mode)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            all_reduce_fusion_config=args.all_reduce_fusion_config,
        )
    else:
        device_num = None
        rank_id = None

    set_seed(args.seed)

    # dataset
    dataset = create_segment_dataset(
            mage_mean=args.image_mean,
            image_std=args.image_std,
            data_file=args.data_dir,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            max_scale=args.max_scale,
            min_scale=args.min_scale,
            ignore_label=args.ignore_label,
            num_classes=args.num_classes,
            num_readers=args.num_readers,
            num_parallel_calls=args.num_parallel_calls,
            shard_id=args.shard_id,
            shard_num=args.shard_num,
            shuffle=True,
            is_training=True 
    )
    steps_per_epoch = dataset.get_dataset_size()

    # use mindcv models as backbone for DeeplabV3
    backbone = create_model(
        args.backbone,
        checkpoint_path=args.backbone_ckpt_path,
        auto_mapping=args.get("backbone_ckpt_auto_mapping", False),
        features_only=args.backbone_features_only,
        out_indices=args.backbone_out_indices,
        output_strip= args.output_strip,
    )

    # network
    deeplabv3 = DeepLabV3(backbone, args, is_training=True)
    ms.amp.auto_mixed_precision(deeplabv3, amp_level=args.amp_level)

    # loss
    loss = SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    model = DeepLabV3WithLoss(deeplabv3, loss)

    # learning rate schedule
    lr_scheduler = create_segment_lr_scheduler(args, steps_per_epoch)
    



    


    