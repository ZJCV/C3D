# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: trainer.py
@author: zj
@description: 
"""

import os
import torch
import argparse

from c3d.data.build import build_dataloader
from c3d.model.build import build_model, build_criterion
from c3d.optim.build import build_optimizer, build_lr_scheduler
from c3d.engine.trainer import do_train
from c3d.engine.inference import do_evaluation
from c3d.util.checkpoint import CheckPointer
from c3d.util.logger import setup_logger
from c3d.util.collect_env import collect_env_info
from c3d.config import cfg


def train(cfg, device=None):
    output_dir = cfg.OUTPUT.DIR
    logger = setup_logger(cfg.TRAIN.NAME, save_dir=output_dir)

    data_loader = build_dataloader(cfg, train=True)
    criterion = build_criterion()
    model = build_model(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    arguments = {"iteration": 0}
    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=output_dir,
                                save_to_disk=True, logger=logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    model = do_train(cfg, arguments,
                     model, criterion, optimizer, lr_scheduler, data_loader,
                     checkpointer, logger, device=device)
    return model


def main():
    parser = argparse.ArgumentParser(description='C3D Training With PyTorch')
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=bool)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.TRAIN.LOG_STEP = args.log_step
    cfg.TRAIN.SAVE_STEP = args.save_step
    cfg.TRAIN.EVAL_STEP = args.eval_step
    cfg.freeze()

    output_dir = cfg.OUTPUT.DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("C3D", save_dir=cfg.OUTPUT.DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = train(cfg, device)

    logger.info('Start final evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model, device)


if __name__ == '__main__':
    main()
