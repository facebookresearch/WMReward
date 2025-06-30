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


_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_path,
    property,
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
    dataset = GRASPDataset(
        data_path=data_path,
        property=property,
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



class GRASPDataset(torch.utils.data.Dataset):
    """ IntPhys dataset """

    def __init__(
        self,
        data_path,
        property="ObjectPermanence",
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

    
        self.scenes = sorted(os.listdir(self.data_path))
        self.scenes = sorted(os.listdir(f"{self.data_path}/P_{self.property}"))
        self.length_clip = self.frames_per_clip*self.frame_step


    def __getitem__(self, index):
        scene = self.scenes[index]
                             
        possible_path = f"{self.data_path}/P_{self.property}/{scene}"
        impossible_path = f"{self.data_path}/IP_{self.property}/{scene}"
        labels = torch.Tensor([1.,0.])
        videos = []
        for vid_path in [possible_path,impossible_path]:
            vr = VideoReader(vid_path, num_threads=-1, ctx=cpu(0))
            vr.seek(0)
            frames = vr.get_batch(np.arange(len(vr))[::self.frame_step]).asnumpy()
            if self.transform:
                frames = self.transform(frames)
            else:
                frames = torch.Tensor(frames)
            videos.append(frames)              
            
        buffer = torch.stack(videos) 
        return buffer,labels



    def __len__(self):
        return len(self.scenes)
