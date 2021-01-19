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

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))



def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .dataset_wrappers import ConcatDataset
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)

    print("The split value is", split)

    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    from .dataset_wrappers import ConcatDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(cfg, dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',al=False, is_train= False, labeled=True,
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
    if al and is_train and labeled:
        shuffle = False


        labeled = os.path.join(cfg.data.train.data_root,"labeled.txt")
        unlabeled = os.path.join(cfg.data.train.data_root,"unlabeled.txt")

        image_set = []
        for sample in dataset:
            image_set.append(sample['img_metas'].data['ori_filename'])


        labeled_reader = csv.reader(open(labeled, 'rt'))
        labeled_image_set = [r[0] for r in labeled_reader]
        unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]


        #print(image_set)
        #print(labeled_image_set)
        #print(unlabeled_image_set)


        labeled_indices = []
        for sample in labeled_image_set:
            labeled_indices.append(image_set.index(sample))
        #print(indices)
        unlabeled_indices = []
        for sample in labeled_image_set:
            unlabeled_indices.append(image_set.index(sample))


        rank, world_size = get_dist_info()
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


        rank, world_size = get_dist_info()
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

    else:
        rank, world_size = get_dist_info()
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


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
