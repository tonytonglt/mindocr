'''
Model training
'''
import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.arg_parser import parse_args

args = parse_args()

# modelarts
from tools.modelarts_adapter.modelarts import modelarts_setup

modelarts_setup(args)  # setup env for modelarts platform if enabled

import yaml
from addict import Dict
import cv2

import mindspore as ms
from mindspore import nn
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import LossMonitor, TimeMonitor

from mindocr.optim import create_optimizer, create_group_params  # adopted from mindcv
from mindocr.scheduler import create_scheduler  # adopted from mindcv

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.losses import build_loss
from mindocr.postprocess import build_postprocess
from mindocr.metrics import build_metric
from mindocr.utils.logger import get_logger
from mindocr.utils.model_wrapper import NetWithLossWrapper
from mindocr.utils.train_step_wrapper import TrainOneStepWrapper
from mindocr.utils.callbacks import EvalSaveCallback
from mindocr.utils.seed import set_seed
from mindocr.utils.loss_scaler import get_loss_scales


def main(cfg):
    # init env
    ms.set_context(mode=cfg.system.mode, device_id=7)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True,
                                     # parameter_broadcast=True,
                                     )
    else:
        device_num = None
        rank_id = None

    set_seed(cfg.system.seed)

    is_main_device = rank_id in [None, 0]

    # create logger, only rank0 log will be output to the screen
    logger = get_logger(log_dir=cfg.train.ckpt_save_dir, rank=rank_id if rank_id else 0, is_main_device=is_main_device)

    # create dataset
    loader_train = build_dataset(
        cfg.train.dataset,
        cfg.train.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=True,
    )
    num_batches = loader_train.get_dataset_size()

    loader_eval = None
    if cfg.system.val_while_train:
        loader_eval = build_dataset(
            cfg.eval.dataset,
            cfg.eval.loader,
            num_shards=device_num,
            shard_id=rank_id,
            is_train=False,
            refine_batch_size=True,
            )

    # create model
    network = build_model(cfg.model)
    amp_level = cfg.system.get('amp_level', 'O0')
    ms.amp.auto_mixed_precision(network, amp_level=amp_level)

    # create loss
    loss_fn = build_loss(cfg.loss.pop('name'), **cfg['loss'])

    net_with_loss = NetWithLossWrapper(network, loss_fn, 
                                       pred_cast_fp32=(amp_level!='O0'),
                                    )  # wrap train-one-step cell

    # get loss scale setting for mixed precision training
    loss_scale_manager, optimizer_loss_scale = get_loss_scales(cfg)

    # build lr scheduler
    lr_scheduler = create_scheduler(num_batches, **cfg['scheduler'])

    # build optimizer
    cfg.optimizer.update({'lr': lr_scheduler, 'loss_scale': optimizer_loss_scale})
    params = create_group_params(network.trainable_params(), **cfg.optimizer)
    optimizer = create_optimizer(params, **cfg.optimizer)

    # build train step cell
    gradient_accumulation_steps = cfg.train.get('gradient_accumulation_steps', 1)
    clip_grad = cfg.train.get('clip_grad', False)
    train_net = TrainOneStepWrapper(net_with_loss,
                                    optimizer=optimizer,
                                    scale_sense=loss_scale_manager,
                                    drop_overflow_update=cfg.system.drop_overflow_update,
                                    gradient_accumulation_steps=gradient_accumulation_steps,
                                    clip_grad=clip_grad,
                                    clip_norm=cfg.train.get('clip_norm', 1.0),
                                    )
    # build postprocess and metric
    postprocessor = None
    metric = None
    if cfg.system.val_while_train:
        # postprocess network prediction
        postprocessor = build_postprocess(cfg.postprocess)
        metric = build_metric(cfg.metric, device_num=device_num)

    # build callbacks
    eval_cb = EvalSaveCallback(
        network,
        loader_eval,
        postprocessor=postprocessor,
        metrics=[metric],
        pred_cast_fp32=(amp_level!='O0'),
        rank_id=rank_id,
        logger=logger,
        batch_size=cfg.train.loader.batch_size,
        ckpt_save_dir=cfg.train.ckpt_save_dir,
        main_indicator=cfg.metric.main_indicator,
        num_columns_to_net=cfg.eval.dataset.get('num_columns_to_net', 1),
        num_columns_of_labels=cfg.eval.dataset.get('num_columns_of_labels', None),
        val_interval=cfg.system.get('val_interval', 1),
        val_start_epoch=cfg.system.get('val_start_epoch', 1),
        log_interval=cfg.system.get('log_interval', 100)
    )

    # log
    num_devices = device_num if device_num is not None else 1
    global_batch_size = cfg.train.loader.batch_size * num_devices * gradient_accumulation_steps
    logger.info('=' * 40)
    logger.info(
        f'Num devices: {num_devices}\n'
        f'Batch size: {cfg.train.loader.batch_size}\n'
        f'Global batch size: {cfg.train.loader.batch_size}x{num_devices}x{gradient_accumulation_steps}={global_batch_size}\n'
        f'Optimizer: {cfg.optimizer.opt}\n'
        f'Scheduler: {cfg.scheduler.scheduler}\n'
        f'LR: {cfg.scheduler.lr} \n'
        f'Auto mixed precision: {cfg.system.amp_level}\n'
        f'Loss scale setting: {cfg.loss_scaler}\n'
        f'drop_overflow_update: {cfg.system.drop_overflow_update}\n'
        f'clip_grad: {clip_grad}\n'
        f'Num epochs: {cfg.scheduler.num_epochs}\n'
        f'Num steps per epoch: {num_batches}'
    )
    if 'name' in cfg.model:
        logger.info(f'Model: {cfg.model.name}')
    else:
        logger.info(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
    logger.info('=' * 40)

    # save args used for training
    # FIXME: some values are popped and not saved.
    if is_main_device:
        with open(os.path.join(cfg.train.ckpt_save_dir, 'args.yaml'), 'w') as f:
            args_text = yaml.safe_dump(cfg.to_dict(), default_flow_style=False, sort_keys=False)
            f.write(args_text)

    # training
    model = ms.Model(train_net)
    model.train(cfg.scheduler.num_epochs, loader_train, callbacks=[eval_cb],
                dataset_sink_mode=cfg.train.dataset_sink_mode)


if __name__ == '__main__':
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)

    if args.enable_modelarts:
        import moxing as mox
        from tools.modelarts_adapter.modelarts import get_device_id, sync_data

        dataset_root = '/cache/data/'
        # download dataset from server to local on device 0, other devices will wait until data sync finished.
        sync_data(args.data_url, dataset_root)
        if get_device_id() == 0:
            # mox.file.copy_parallel(src_url=args.data_url, dst_url=dataset_root)
            print(f'INFO: datasets found: {os.listdir(dataset_root)} \n'
                  f'INFO: dataset_root is changed to {dataset_root}'
                  )
        # update dataset root dir to cache
        assert 'dataset_root' in config['train'][
            'dataset'], f'`dataset_root` must be provided in the yaml file for training on ModelArts or OpenI, but not found in {yaml_fp}. Please add `dataset_root` to `train:dataset` and `eval:dataset` in the yaml file'
        config.train.dataset.dataset_root = dataset_root
        config.eval.dataset.dataset_root = dataset_root

    # main train and eval
    main(config)

    if args.enable_modelarts:
        # upload models from local to server
        if get_device_id() == 0:
            mox.file.copy_parallel(src_url=config.train.ckpt_save_dir, dst_url=args.train_url)
