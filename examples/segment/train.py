import argparse
import os

# import sys

import yaml
import mindspore as ms
from addict import Dict
from mindspore.communication import get_group_size, get_rank, init
from mindspore import log as logger
from mindspore import save_checkpoint
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)

from data import create_segment_dataset
from loss import SoftmaxCrossEntropyLoss
from deeplabv3 import DeepLabV3
from deeplab_resnet import *

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
# from config import parse_args


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
        name=args.dataset,
        image_mean=args.image_mean,
        image_std=args.image_std,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        max_scale=args.max_scale,
        min_scale=args.min_scale,
        ignore_label=args.ignore_label,
        num_classes=args.num_classes,
        num_parallel_workers=args.num_parallel_workers,
        # num_readers=args.num_readers,
        # num_parallel_calls=args.num_parallel_calls,
        shard_id=rank_id,
        shard_num=device_num,
        shuffle=True,
        is_training=True,
    )

    steps_per_epoch = dataset.get_dataset_size()

    # use mindcv models as backbone for DeeplabV3
    backbone = create_model(
        args.backbone,
        checkpoint_path=args.backbone_ckpt_path,
        auto_mapping=args.get("backbone_ckpt_auto_mapping", False),
        features_only=args.backbone_features_only,
        out_indices=args.backbone_out_indices,
        output_stride=args.output_stride,
    )

    # network
    # just network no loss
    # loss used in create_train in mindcv
    deeplabv3 = DeepLabV3(backbone, args, is_training=True)
    ms.amp.auto_mixed_precision(deeplabv3, amp_level=args.amp_level)

    # loss
    loss = SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    # network = DeepLabV3WithLoss(deeplabv3, loss)

    # learning rate schedule
    lr_scheduler = create_scheduler(
        steps_per_epoch,
        scheduler=args.scheduler,  # poly cosine exp
        lr=args.lr,
        min_lr=args.min_lr,
        # warmup_epochs=args.warmup_epochs,
        # warmup_factor=args.warmup_factorx,
        decay_epochs=args.decay_epochs,
        decay_rate=args.decay_rate,
        milestones=args.multi_step_decay_milestones,
        num_epochs=args.epoch_size,
        # num_cycles=args.num_cycles,
        # cycle_decay=args.cycle_decay,
        lr_epoch_stair=args.lr_epoch_stair,
    )

    # resume training if ckpt_path is given
    if args.ckpt_path != "" and args.resume_opt:
        opt_ckpt_path = os.path.join(args.ckpt_save_dir, f"optim_{args.model}.ckpt")
    else:
        opt_ckpt_path = ""

    # create optimizer
    if (
        args.loss_scale_type == "fixed"
        and args.drop_overflow_update is False
        and not require_customized_train_step(
            args.ema,
            args.clip_grad,
            args.gradient_accumulation_steps,
            # args.amp_cast_list,
        )
    ):
        optimizer_loss_scale = args.loss_scale
    else:
        optimizer_loss_scale = 1.0

    optimizer = create_optimizer(
        deeplabv3.trainable_params(),
        opt="momentum",
        lr=lr_scheduler,
        weight_decay=args.weight_decay,  # 0.0001
        momentum=args.momentum,  # 0.9
        filter_bias_and_bn=args.filter_bias_and_bn,
        loss_scale=1.0, # TODO: loss_scale disabled to avoid NPU ops
        checkpoint_path=opt_ckpt_path,
    )

    # TODO: define eval metrics ?

    # create trainer
    # 存疑  metrics=None
    trainer = create_trainer(
        deeplabv3,
        loss,
        optimizer,
        metrics=None,
        amp_level = "O0" if args.device_target == "CPU" else args.amp_level,
        amp_cast_list=args.amp_cast_list,
        loss_scale_type=args.loss_scale_type,
        loss_scale=args.loss_scale,  # fixed
        drop_overflow_update=args.drop_overflow_update,  # False
    )

    # callback
    callbacks = [TimeMonitor(data_size=steps_per_epoch), LossMonitor()]

    if rank_id == 0:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=args.save_steps,
            keep_checkpoint_max=args.keep_checkpoint_max,
        )
        prefix_name = "deeplabv3_s" + str(args.output_stride) + "_" + args.backbone
        ckpt_cb = ModelCheckpoint(
            prefix=prefix_name, directory=args.ckpt_save_dir, config=ckpt_config
        )
        callbacks.append(ckpt_cb)

    # if args.eval_while_train and rank_id==0:
    #     eval_model =
    #     eval_dataset = create_segment_dataset()
    #     eval_callback =
    #     callbacks.append(eval_callback)

    trainer.train(
        args.epoch_size,
        dataset,
        callbacks=callbacks,
        dataset_sink_mode=(args.device_target != "CPU"),
    )
    # model.train(args.train_epochs, dataset, callbacks=cbs, dataset_sink_mode=(args.device_target != "CPU"))


def parse_args():
    parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        help="YAML config file specifying default arguments (default=" ")",
    )
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Accumulate the gradients of n batches before update.",
    # )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    from mindspore import context
    context.set_context(device_target=args.device_target) 

    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        args = yaml.safe_load(fp)
    args = Dict(args)
    
    # data sync for cloud platform if enabled
    if args.enable_modelarts:
        import moxing as mox

        args.data_dir = f"/cache/{args.data_url}"
        mox.file.copy_parallel(
            src_url=os.path.join(args.data_url, args.dataset), dst_url=args.data_dir
        )

    # core training
    train(args)

    if args.enable_modelarts:
        mox.file.copy_parallel(src_url=args.ckpt_save_dir, dst_url=args.train_url)
