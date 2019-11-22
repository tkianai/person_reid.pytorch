import sys
import os
import os.path as osp
import time
import argparse

import torch
import torch.nn as nn

from reid.config import (
    get_default_config, imagedata_kwargs, optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs, get_defeault_exp_name
)
from reid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)
from reid.data import ImageDataManager
from reid.optim import build_optimizers, build_lr_schedulers
from reid.engine import ImageEngine
from reid.modeling import build_model


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms
    if args.test:
        cfg.test.evaluate = args.test

    if not args.name:
        args.name = get_defeault_exp_name(cfg)
    cfg.data.save_dir = osp.join(cfg.data.save_dir, args.name)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str,
                        default='', help='path to config file')
    parser.add_argument('--name', default=None, help='identify exp name')
    parser.add_argument('--test', type=bool, default=False, help='train or test')
    parser.add_argument('-s', '--sources', type=str, nargs='+',
                        help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+',
                        help='target datasets (delimited by space)')
    parser.add_argument('--transforms', type=str,
                        nargs='+', help='data augmentation')
    parser.add_argument('--root', type=str, default='',
                        help='path to data root')
    parser.add_argument('--gpu-devices', type=str, default='',)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    if cfg.use_gpu and args.gpu_devices:
        # if gpu_devices is not specified, all available gpus will be used
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = ImageDataManager(**imagedata_kwargs(cfg))

    print('Building model: {}'.format(cfg.model.backbone.name))
    model = build_model(
        cfg,
        num_classes=datamanager.num_train_pids,
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        if 'CUDA_VISIBLE_DEVICES' not in os.environ or len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = nn.DataParallel(model)
        model = model.cuda()

    optimizers = build_optimizers(model, **optimizer_kwargs(cfg))
    schedulers = build_lr_schedulers(optimizers, *lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizers=optimizers)

    print('Building {}-engine'.format(cfg.loss.name))
    engine = ImageEngine(
        datamanager, model, optimizers, optimizer_weights=cfg.solver.optim_weights, schedulers=schedulers)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
