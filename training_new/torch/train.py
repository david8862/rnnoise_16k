#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
/* Copyright (c) 2024 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""
import os, sys, argparse, time
import glob
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from rnnoise.model import RNNoise
from rnnoise.data import get_dataloader, RNNoiseDataset, mask
from common.model_utils import get_optimizer, get_lr_scheduler


# global value to record the best loss
best_loss = float("inf")


def checkpoint_clean(checkpoint_dir, max_keep=5):
    # filter out checkpoints and sort
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'ep*.pth')), reverse=False)

    # keep latest checkpoints
    for checkpoint in checkpoints[:-(max_keep)]:
        os.remove(checkpoint)



def train(args):
    global best_loss

    log_dir = 'logs/000'
    os.makedirs(log_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # prepare dataset loader
    dataloader = get_dataloader(args.feature_file, args.bands_num, args.delta_ceps_num, args.sequence_length, args.batch_size, use_cuda)

    # get tensorboard summary writer
    summary_writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

    #model = RNNoise(cond_size=args.cond_size, gru_size=args.gru_size)
    model = RNNoise(bands_num=args.bands_num, delta_ceps_num=args.delta_ceps_num, cond_size=args.cond_size, gru_size=args.gru_size)

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
        print('Load weights {}.'.format(args.weights_path))

    optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay)

    # learning rate scheduler
    lr_scheduler = get_lr_scheduler(args.decay_type, optimizer)

    model.to(device)
    summary(model, input_size=(2000, 38))

    states = None
    for epoch in range(1, args.total_epoch+1):

        running_gain_loss = 0
        running_vad_loss = 0
        running_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, gain, vad) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                gain = gain.to(device)
                vad = vad.to(device)

                pred_gain, pred_vad, states = model(features, states=states)
                states = [state.detach() for state in states]
                gain = gain[:,3:-1,:]
                vad = vad[:,3:-1,:]
                target_gain = torch.clamp(gain, min=0)
                target_gain = target_gain*(torch.tanh(8*target_gain)**2)

                e = pred_gain**args.gamma - target_gain**args.gamma
                gain_loss = torch.mean((1+5.*vad)*mask(gain)*(e**2))
                #vad_loss = torch.mean(torch.abs(2*vad-1)*(vad-pred_vad)**2)
                vad_loss = torch.mean(torch.abs(2*vad-1)*(-vad*torch.log(.01+pred_vad) - (1-vad)*torch.log(1.01-pred_vad)))
                loss = gain_loss + .001*vad_loss

                loss.backward()
                optimizer.step()
                if args.sparse:
                    model.sparsify()

                lr_scheduler.step()

                running_gain_loss += gain_loss.detach().cpu().item()
                running_vad_loss += vad_loss.detach().cpu().item()
                running_loss += loss.detach().cpu().item()
                tepoch.set_postfix(loss=f"{running_loss/(i+1):8.5f}",
                                   gain_loss=f"{running_gain_loss/(i+1):8.5f}",
                                   vad_loss=f"{running_vad_loss/(i+1):8.5f}",
                                   )

        # save checkpoint with best loss
        if running_loss < best_loss:
            os.makedirs(log_dir, exist_ok=True)
            checkpoint_dir = os.path.join(log_dir, 'ep{epoch:03d}-loss{running_loss:.3f}-gain_loss{running_gain_loss:.3f}-vad_loss{running_vad_loss:.3f}.pth'.format(epoch=epoch, running_loss=running_loss, running_gain_loss=running_gain_loss, running_vad_loss=running_vad_loss))
            torch.save(model, checkpoint_dir)
            print('Epoch {epoch:03d}: best_loss improved from {best_loss:.3f} to {running_loss:.3f}, saving model to {checkpoint_dir}'.format(epoch=epoch, best_loss=best_loss, running_loss=running_loss, checkpoint_dir=checkpoint_dir))
            best_loss = running_loss
        else:
            print('Epoch {epoch:03d}: best_loss did not improve from {best_loss:.3f}'.format(epoch=epoch, best_loss=best_loss))

        checkpoint_clean(log_dir, max_keep=5)

    # Finally store model
    torch.save(model, os.path.join(log_dir, 'trained_final.pth'))
    #torch.save(model.state_dict(), os.path.join(log_dir, 'trained_final.pt'))



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='train RNNoise model')
    # Model definition options
    # input feature number = bands_num + 3*delta_ceps_num + 2
    # output feature number = bands_num
    parser.add_argument('--bands_num', type=int, required=False, default=18,
        help="number of bands, default=%(default)s")
    parser.add_argument('--delta_ceps_num', type=int, required=False, default=6,
        help="number of delta ceps, default=%(default)s")
    parser.add_argument('--cond_size', type=int, required=False, default=128,
        help="first conditioning size, default=%(default)s")
    parser.add_argument('--gru_size', type=int, required=False, default=384,
        help="gru layer size, default=%(default)s")
    parser.add_argument('--sequence_length', type=int, required=False, default=2000,
        help="input sequence length, default=%(default)s")
    parser.add_argument('--sparse', default=False, action="store_true",
        help='Whether to sparsify the gru layers during training')
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help="pretrained model/weights file for fine tune, default=%(default)s")

    # Data options
    parser.add_argument('--feature_file', type=str, required=True,
        help='audio feature file in .f32 format for training')
    #parser.add_argument('--val_split', type=float, required=False, default=0.1,
    #    help="validation data persentage in dataset, default=%(default)s")

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=128,
        help="batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adamw', choices=['adam', 'adamw', 'rmsprop', 'sgd'],
        help="optimizer for training (adam/adamw/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help="Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default='lambda', choices=[None, 'lambda', 'exponential', 'polynomial', 'piecewise_constant'],
        help="Learning rate decay type, default=%(default)s")
    parser.add_argument('--weight_decay', type=float, required=False, default=5e-4,
        help="Weight decay for optimizer, default=%(default)s")
    parser.add_argument('--gamma', type=float, required=False, default=0.25,
        help="perceptual exponent, default=%(default)s")

    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help="Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--transfer_epoch', type=int, required=False, default=5,
        help="Transfer training (from Imagenet) stage epochs, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=100,
        help="Total training epochs, default=%(default)s")
    #parser.add_argument('--gpu_num', type=int, required=False, default=1,
    #    help='Number of GPU to use, default=%(default)s')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    train(args)



if __name__ == "__main__":
    main()
