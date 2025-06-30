# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from PIL import Image
import torch
import json
from decord import VideoReader,cpu
import pandas as pd


_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_path,
    property,
    priming,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    deterministic=True,
    log_dir=None,
):
    dataset = InfLevelDataset(
        data_path=data_path,
        property=property,
        priming=priming,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('Dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=drop_last)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True)

    logger.info('Intuitive Physics data loader created')

    return dataset, data_loader, dist_sampler



class InfLevelDataset(torch.utils.data.Dataset):
    """ IntPhys dataset """

    def __init__(
        self,
        data_path,
        property="continuity",
        priming=False,
        frames_per_clip=16, # Unused
        frame_step=10,
        transform=None,
        shared_transform=None,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.shared_transform = shared_transform
        self.property = property
        self.priming = priming
        
        self.metadatas = pd.read_csv(f"{os.path.abspath(os.path.dirname(__file__))}/../../auxiliary_data_loading_files/inflevel/{self.property}.csv")

        self.length_clip = self.frames_per_clip*self.frame_step
        


    def __getitem__(self, index):
        datas = self.metadatas.iloc[index]

        labels = []
        videos = []

        #Some videos are a few frames off, so we want to avoid any errors here
        # We remove the first frame in that scenario, before the sampling with frame_step
        # This doubles the code but is the simplest option whithout opening files multiple times

        vr1 = VideoReader(self.data_path + '/' + datas[f'vid1_path'], num_threads=-1, ctx=cpu(0))
        vr1.seek(0)

        vr2 = VideoReader(self.data_path + '/' + datas[f'vid2_path'], num_threads=-1, ctx=cpu(0))
        vr2.seek(0)
        frames_all = []
        if self.priming:
            diff = len(vr1) - len(vr2)
            if diff > 0:
                frames_all.append(vr1.get_batch(np.arange(len(vr1))[diff::self.frame_step]).asnumpy())
            else:
                frames_all.append(vr1.get_batch(np.arange(len(vr1))[::self.frame_step]).asnumpy())

            if diff < 0:
                frames_all.append(vr2.get_batch(np.arange(len(vr2))[-diff::self.frame_step]).asnumpy())
            else:
                frames_all.append(vr2.get_batch(np.arange(len(vr2))[::self.frame_step]).asnumpy())
        else:
            diff = (len(vr1)-datas[f'vid1_end_priming_2']) - (len(vr2)-datas[f'vid2_end_priming_2'])
            if diff > 0:
                frames_all.append(vr1.get_batch(np.arange(len(vr1))[datas[f'vid1_end_priming_2']+diff+1::self.frame_step]).asnumpy())
            else:
                frames_all.append(vr1.get_batch(np.arange(len(vr1))[datas[f'vid1_end_priming_2']+1::self.frame_step]).asnumpy())

            if diff < 0:
                frames_all.append(vr2.get_batch(np.arange(len(vr2))[datas[f'vid2_end_priming_2']-diff+1::self.frame_step]).asnumpy())
            else:
                frames_all.append(vr2.get_batch(np.arange(len(vr2))[datas[f'vid2_end_priming_2']+1::self.frame_step]).asnumpy())


        for frames in frames_all:
            if self.transform:
                frames = self.transform(frames)
            else:
                frames = torch.Tensor(frames)
            videos.append(frames)          

        if datas[f'vid1_label'] == 'possible':
            labels = torch.Tensor([1.0,0.0])
        else:
            labels = torch.Tensor([0.0,1.0])

        buffer = torch.stack(videos) 
        return buffer,labels



    def __len__(self):
        return len(self.metadatas)
