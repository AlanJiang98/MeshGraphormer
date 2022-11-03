"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os.path as op
import torch
import logging
import code
from src.utils.comm import get_world_size, is_main_process
from src.datasets.Freihand0 import FreiHand0
from src.datasets.EvRealHands import EvRealHands
from src.datasets.EvRealHands_web import EvRealHandsWeb
from src.datasets.Interhand import Interhand
from src.datasets.Interhand_web import InterhandWeb
from torch.utils.data import ConcatDataset
import pdb
from tqdm import tqdm, trange


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


#==============================================================================================


def build_hand_dataset(config, is_train=True):
    '''
    generate dataset class for different datasets
    '''
    datasets = []
    datasets_test = []
    if 'freihand0' in config['data']['dataset']:
        yaml_file = config['data']['train_yaml'] if is_train else config['data']['eval_yaml']
        print(yaml_file)
        datasets.append(FreiHand0(config, yaml_file, is_train, False, config['data']['img_scale_factor']))
    if 'freihand1' in config['data']['dataset']:
        pass
    if 'interhand' in config['data']['dataset']:
        if is_main_process():
            print(50*'*')
            print('start to load interhand!')
        datasets.append(Interhand(config))
        datasets_test.append(InterhandWeb(config))
    if 'evrealhands' in config['data']['dataset']:
        if is_main_process():
            print(50*'*')
            print('start to load evrealhands!')
        datasets.append(EvRealHands(config))
        datasets_test.append(EvRealHandsWeb(config))
    return ConcatDataset(datasets), ConcatDataset(datasets_test)


def make_hand_data_loader(config, start_iter=0):
    '''
    generate distributed dataloader
    '''
    is_train = False if config['exper']['run_eval_only']==True else True
    dataset, dataset_test = build_hand_dataset(config, is_train=is_train)
    for i in trange(len(dataset)):
        result1 = dataset.__getitem__(i)
        result2 = dataset_test.__getitem__(i)
        # pdb.set_trace()
        for j in range(len(result1)):
            if j ==1:
                if str(result1[j][0]) == str(result2[j][0]):
                    # pdb.set_trace()
                    continue
                else:
                    print('error')
                    pdb.set_trace()
            for k in range(len(result1[j])):
                
                for key in result1[j][k].keys():
                    if key == 'delta_time': #or key == "ev_frames":
                        continue
                    if type(result1[j][k][key]) in [float, int]:
                        continue
                    if len(result1[j][k][key])>=1 and len(result2[j][k][key]) == len(result1[j][k][key]):
                        for step in range(len(result1[j][k][key])):
                            if not len(result2[j][k][key]) == len(result1[j][k][key]):
                                print("error len")
                                pdb.set_trace()
                            if not (result1[j][k][key][step] == result2[j][k][key][step]).all():
                                print('error 1')
                                pdb.set_trace()
                    else:
                        print("error shape")
                        pdb.set_trace()
    print('check done!')
    pdb.set_trace()
    
    if is_train==True:
        shuffle = True
        images_per_gpu = config['exper']['per_gpu_batch_size']
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * config['exper']['num_train_epochs']
        if is_main_process():
            print(50*'*')
            print("Train with {} images per GPU.".format(images_per_gpu))
            print("Total batch size {}".format(images_per_batch))
            print("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = config['exper']['per_gpu_batch_size']
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, config['exper']['distributed'])
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=config['exper']['num_workers'], batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader

