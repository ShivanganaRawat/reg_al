#train
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg.apis.train import set_random_seed
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

#test
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

#trainsegmentor
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataset
from mmseg.utils import get_root_logger

#build dataloader
import copy
import platform
import random
from functools import partial

import numpy as np
import csv
import os
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import SubsetRandomSampler




def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor using active learning')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)


    parser.add_argument('--al', default=False, help='Train with active learning', action='store_true')
    parser.add_argument('--batch_size', type=int, help='Batch size for active learning')
    parser.add_argument('--max_episode', type=int, help='Maximum active learning episodes')

    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args



def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',cfg=None, al=True, is_train = False, labeled=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        dataloader_type (str): Type of dataloader. Default: 'PoolDataLoader'
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    #print(cfg.data)
    rank, world_size = get_dist_info()
    if al and is_train and labeled:
        shuffle = False


        labeled = os.path.join(cfg.data.train.data_root,"labeled.txt")
        #unlabeled = os.path.join(cfg.data.train.data_root,"unlabeled.txt")

        image_set = []
        for sample in dataset:
            image_set.append(sample['img_metas'].data['ori_filename'])



        labeled_reader = csv.reader(open(labeled, 'rt'))
        labeled_image_set = [r[0] for r in labeled_reader]


        #unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
        #unlabeled_image_set = [r[0] for r in unlabeled_reader]


        #print(image_set)
        #print(labeled_image_set)
        #print(unlabeled_image_set)


        labeled_indices = []
        for sample in labeled_image_set:
            labeled_indices.append(image_set.index(sample))

        #unlabeled_indices = []
        #for sample in labeled_image_set:
        #    unlabeled_indices.append(image_set.index(sample))

        if dist:
            samp = SubsetRandomSampler(labeled_indices)
            sampler = DistributedSamplerWrapper(samp, world_size, rank, shuffle=shuffle) 
            #DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
            shuffle = False
            batch_size = samples_per_gpu
            num_workers = workers_per_gpu
        else:
            sampler = SubsetRandomSampler(labeled_indices)
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu


    elif al and is_train:
        shuffle = False


        unlabeled = os.path.join(cfg.data.train.data_root,"unlabeled.txt")

        image_set = []
        for sample in dataset:
            image_set.append(sample['img_metas'].data['ori_filename'])


        unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]

        unlabeled_indices = []
        for sample in unlabeled_image_set:
            unlabeled_indices.append(image_set.index(sample))


        if dist:
            samp = SubsetRandomSampler(unlabeled_indices)
            sampler = DistributedSamplerWrapper(samp, world_size, rank, shuffle=shuffle) 
            #DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
            shuffle = False
            batch_size = samples_per_gpu
            num_workers = workers_per_gpu
        else:
            sampler = SubsetRandomSampler(unlabeled_indices)
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu

        
    else:
        if dist:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle)
            shuffle = False
            batch_size = samples_per_gpu
            num_workers = workers_per_gpu
        else:
            sampler = None
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    assert dataloader_type in (
        'DataLoader',
        'PoolDataLoader'), f'unsupported dataloader {dataloader_type}'

    if dataloader_type == 'PoolDataLoader':
        dataloader = PoolDataLoader
    elif dataloader_type == 'DataLoader':
        dataloader = DataLoader

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)


    return data_loader


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None, al=False):

    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)


    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True, al=True, cfg=cfg, is_train = not bool(i), labeled=True) for i, ds in enumerate(dataset)
    ]


    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)


    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False, cfg=cfg, al=True, is_train=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
    

def train(args, cfg, logger, timestamp, episode=0):

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    meta['env_info'] = env_info


    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)



    model = build_segmentor(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    #logger.info(model)
    #print(cfg)

    datasets = [build_dataset(cfg.data.train)]

    #print(datasets[0][0]['gt_semantic_seg'].data.shape)

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            #mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    #print(model)
    #print(datasets)
    #print(cfg)
    
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta, al=True)
    return logger, model


def single_gpu_inference(model, data_loader):
    model.eval()
    #print(type(model))
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            #print(data)
            result = model(return_loss=False, return_logits=True, **data)
            results.append([data['img_metas'].data[0][0]['ori_filename'], result])
            #print(result.shape)
            #result = model.whole_inference(image_data[1]['img'].data[0], image_data[1]['img_metas'].data[0], rescale=True)
            #result = F.softmax(result, dim=1)
            #info_content = torch.sum(torch.mul(torch.sum(torch.mul(result, torch.log(result)), dim=1),-1))
            #print(image_data[1]['img_metas'].data)#['ori_filename'])
            #results.append([image_data[1]['img_metas'].data[0][0]['ori_filename'], info_content])

    return results


def multi_gpu_inference(model, data_loader, tmpdir=None, gpu_collect=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.whole_inference(image_data[1]['img'].data[0], image_data[1]['img_metas'].data[0], rescale=True)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


#def uncertainty_sampling():
    

def test_and_update_pools(args, cfg, model):
    args = parse_args()


    #if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    #    raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    '''
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
        '''
    #cfg.model.pretrained = None
    #cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False, cfg=cfg, al=True, is_train=True, labeled=False)

    # build the model and load checkpoint
    #model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    #checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    #model.CLASSES = checkpoint['meta']['CLASSES']
    #model.PALETTE = checkpoint['meta']['PALETTE']

    

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_inference(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_inference(model, data_loader, args.tmpdir,
                                 args.gpu_collect)


    #print(results)
    results = sorted(results, key= lambda x: x[1])
    results = results[:args.batch_size]


    new_batch = [x[0] for x in results]
    
    labeled = os.path.join(cfg.data.train.data_root,"labeled.txt")
    labeled_reader = csv.reader(open(labeled, 'rt'))
    labeled_image_set = [r[0] for r in labeled_reader]
    new_labeled = labeled_image_set + new_batch
    new_labeled.sort()

    with open(os.path.join(cfg.data.train.data_root,"labeled.txt"), 'a') as f:
        for line in new_labeled:
            f.write(line+"\n")


    unlabeled = os.path.join(cfg.data.train.data_root,"unlabeled.txt")
    unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
    unlabeled_image_set = [r[0] for r in unlabeled_reader]
    new_unlabeled = list(set(unlabeled_image_set) - set(new_batch))
    new_unlabeled.sort()

    with open(os.path.join(cfg.data.train.data_root,"unlabeled.txt"), 'w') as f:
        for line in new_unlabeled:
            f.write(line+"\n")




def activelearn():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')


    _, model = train(args=args, cfg=cfg, logger=logger, timestamp=timestamp)

    test_and_update_pools(args=args, cfg=cfg, model=model)





if __name__ == '__main__':
    activelearn()