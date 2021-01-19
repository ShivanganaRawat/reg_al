import argparse
import copy
import os
import os.path as osp
import time
import csv
import random

import mmcv
import torch
from mmcv.runner import init_dist, load_checkpoint
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

#from mmseg import __version__
from mmseg.apis.train import set_random_seed, train_segmentor
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
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
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    



    unlabeled = os.path.join(cfg.data.train.data_root,"unlabeled.txt")
    unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
    unlabeled_image_set = [r[0] for r in unlabeled_reader]



    random.seed(0)
    initial_batch = random.sample(unlabeled_image_set, args.batch_size)

    with open(os.path.join(cfg.data.train.data_root,"labeled.txt"), 'a') as f:
            for line in initial_batch:
                f.write(line+"\n")

    new_unlabeled = list(set(unlabeled_image_set) - set(initial_batch))

    with open(os.path.join(cfg.data.train.data_root,"unlabeled.txt"), 'w') as f:
        for line in new_unlabeled:
            f.write(line+"\n")

    unlabeled_image_set_size = len(new_unlabeled)



    episode = 0

    



    while unlabeled_image_set_size != 0 and episode != args.max_episode:


        cfg = Config.fromfile(args.config)
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

        # set random seeds
        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, deterministic: '
                        f'{args.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed
        meta['exp_name'] = osp.basename(args.config)



        model = build_segmentor(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        #logger.info(model)

        datasets = [build_dataset(cfg.data.train)]
        print(datasets)

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmseg version, config file content and class names in
            # checkpoints as meta data
            print("getting config")
            cfg.checkpoint_config.meta = dict(
                #mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=cfg.pretty_text,
                CLASSES=datasets[0].CLASSES,
                PALETTE=datasets[0].PALETTE)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES

        
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta, al=args.al)
            

        print("Training done!")
        

        dataset = build_dataset(cfg.data.train)
        data_loader = build_dataloader(cfg,
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False, al=args.al, is_train=True, labeled=False)


        

        # build the model and load checkpoint
        model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model, os.path.join(cfg.work_dir, "latest.pth"), map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

        
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))


        for image_data in enumerate(data_loader):
            with torch.no_grad():
                print(image_data)
                result = model.whole_inference(image_data[1]['img'].data[0], image_data[1]['img_metas'].data[0], rescale=True)
                result = F.softmax(result, dim=1)
                info_content = torch.sum(torch.mul(torch.sum(torch.mul(result, torch.log(result)), dim=1),-1))
                #print(image_data[1]['img_metas'].data)#['ori_filename'])
                results.append([image_data[1]['img_metas'].data[0][0]['ori_filename'], info_content])

        results = sorted(results, key= lambda x: x[1])
        results = results[:args.batch_size]


        new_batch = [x[0] for x in results]
        
        

        with open(os.path.join(cfg.data.train.data_root,"labeled.txt"), 'a') as f:
            for line in new_batch:
                f.write(line+"\n")


        unlabeled = os.path.join(cfg.data.train.data_root,"unlabeled.txt")
        unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]
        new_unlabeled = list(set(unlabeled_image_set) - set(new_batch))

        with open(os.path.join(cfg.data.train.data_root,"unlabeled.txt"), 'w') as f:
            for line in new_unlabeled:
                f.write(line+"\n")

        unlabeled_image_set_size = len(new_unlabeled)



        episode +=1
        


                


        #print(len(results))

        


if __name__ == '__main__':
    main()
