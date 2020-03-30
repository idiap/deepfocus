'''
Code for the PyTorch implementation of
"DeepFocus: a Few-Shot Microscope Slide Auto-Focus using a Sample-invariant CNN-based Sharpness Function"

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,
All rights reserved.

This file is part of DeepFocus.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

from skimage import io
import torch
import torch.nn.functional as F
import numpy as np
from fastai.basics import load_learner, Learner, dataclass, Callback
from fastai.callbacks import TrainingPhase, GeneralScheduler
from fastai.callback import annealing_cos
from fastai.vision import Image, ImageList, get_transforms, rand_crop, ImageImageList
from fastai.basic_data import DataBunch, DatasetType, TensorDataset, DataLoader

@dataclass
class TensorboardLogger(Callback):
    learn: Learner
    run_name: str
    histogram_freq: int = 50
    path: str = None
    num_epoch : int = 0

    def __post_init__(self):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass


def grayloader(path, onedim=False):
    img = np.asarray(io.imread(path, as_gray=False, plugin='imageio')).astype(np.float32)/65536.
    img = torch.Tensor(img)
    if onedim is True:
        img.unsqueeze_(0)
    else:
        img = img.repeat(3, 1, 1)
    return img


class MyImageList(ImageList):
    def open(self, fn):
        im = Image(grayloader(fn))
        return Image(torch.zeros(im.shape))


class MyImageImageList(ImageImageList):
    _label_cls = MyImageList

    def open(self, fn):
        return Image(grayloader(fn))


def loss_with_flag(outputs, labels):
    zero_or_one = (1.0 - labels[:, -1])
    loss_flag = ((outputs[:, -1] - labels[:, -1]) ** 2).mean()
    loss_parameters = F.smooth_l1_loss(outputs, labels)
    loss = (zero_or_one * loss_parameters).mean() + loss_flag
    return loss


def flattenAnneal(learn:Learner, lr:float, n_epochs:int, start_pct:float):
    n = len(learn.data.train_dl)
    anneal_start = int(n*n_epochs*start_pct)
    anneal_end = int(n*n_epochs) - anneal_start
    phases = [TrainingPhase(anneal_start).schedule_hp('lr', lr),
           TrainingPhase(anneal_end).schedule_hp('lr', lr, anneal=annealing_cos)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(n_epochs)


def get_data(bs, size, src):
    pass


def test_unet(learn, picture_input, downsample=8, batch_size=12, picture=False):
    picture_input = torch.from_numpy(picture_input)
    picture_input.unsqueeze_(dim=1)
    picture_input = F.interpolate(picture_input, size=(224,224), mode='bilinear', align_corners=True).float()
    picture_input = torch.cat([picture_input, picture_input, picture_input], dim=1)
    my_dataset = TensorDataset(picture_input, picture_input)  # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)  # create your dataloader
    my_databunch = DataBunch(train_dl =my_dataloader, test_dl=my_dataloader, valid_dl=my_dataloader)
    learn.data = my_databunch
    output = learn.get_preds(ds_type=DatasetType.Valid)[0]

    if picture:
        import matplotlib.pyplot as plt

        idx = 52

        plt.figure()
        aa = picture_input[idx,:,:,:].data.numpy()
        im_out = np.transpose(aa, (1,2,0))
        plt.imshow(im_out)
        plt.title('input')

        plt.figure()
        aa = output[idx,:-1,:,:].data.numpy()
        im_out = np.transpose(aa, (1,2,0))
        plt.imshow(im_out)
        plt.title('output')
        plt.show()

    output = F.interpolate(output, scale_factor=1.0/downsample, mode='nearest')
    return output.data.cpu().numpy()


def get_learner():

    device = torch.device('cuda')
    learn = load_learner(path='/home/adrian/git/Pytorch-UNet/', file='data/05-03-2020-17:37:43_PROJ_9011_LR_0.0001_BS_30_N_unet_resnet_ATT_False_MODEL_resnet34unetanneal_EXPORT_37.pth', device=device)

    return learn

