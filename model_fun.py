

import argparse
import re
import os
import glob
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import main_train as mt
from sklearn.neighbors import NearestNeighbors
from sklearn import aggregation
import main_test as mt1

padding = 0
cnn_channels = 0
ksize = 0
cnn_outchannels = 0
cnn_bn = 0


# CNN implementation
def cnn(cnn_opt):
    rep = cnn_opt.get("kernel", 3)

    res = 0

    temp_param = {ksize: cnn_opt.get("kernel", 3), padding: (rep-1)//2, cnn_bn: cnn_opt.get("bn", True), cnn_depth: cnn_opt.get(
        "depth", 0), cnn_channels: cnn_opt.get("features"), cnn_outchannels: cnn_opt.get("out_nplanes",), chan_in: cnn_opt.get("nplanes_in")}

    cnn_depth = cnn_opt.get("depth", 0)

    cnn_layers = []
    ReLU = nn.ReLU(inplace=True)

    i = 0

    while(i < (temp_param[cnn_depth]-1)):
        temp_param[cnn_layers].extend([
            nn.Conv2d(temp_param[chan_in], temp_param[cnn_channels], temp_param[ksize],
                      1, temp_param[padding], bias=(not cnn_bn)),
            nn.BatchNorm2d(cnn_channels) if cnn_bn else nn.Sequential(),
            ReLU
        ])
        chan_in = temp_param[cnn_channels]
        i += 1

    if cnn_depth <= 0:
        res = 1
    else:
        val = nn.Conv2d(temp_param[chan_in], temp_param[cnn_outchannels],
                        temp_param[ksize], 1, temp_param[padding], bias=True)
        cnn_layers.append(val)

    net = nn.Sequential(*cnn_layers)
    net.out_nplanes = temp_param[cnn_outchannels]
    print(res)
    net.nplanes_in = cnn_opt.get("nplanes_in")
    res += 1
    return net


res = padding
# Embedding network + KNN selection rule is the neural_nearest_neighbourblock,


class neural_nearest_neighbourblock(nn.Module):

    def __init__(self, nplanes_in, k, patchsize=10, stride=5, nl_match_window=15, temp_opt={}, embedcnn_opt={}):

        super().__init__()

        self.patchsize = patchsize
        self.row = patchsize
        self.stride = stride
        self.row_1 = self.row
        self.nplanes_in = nplanes_in
        self.stride1 = self.stride
        self.out_nplanes = (k+1) * nplanes_in
        self.k1 = k
        self.k = k
        self.reset_parameters()

        # Call embedding network using cnn as:
        embedcnn_opt["nplanes_in"] = nplanes_in
        self.embedcnn = cnn(embedcnn_opt)

        # Temperature tensor

        if temp_opt.get("external_temp"):
            tempcnn_opt = dict(**embedcnn_opt)
            self.k1 = 0
            tempcnn_opt["out_nplanes"] = 1
            self.tempcnn = cnn(tempcnn_opt)
        elif(temp_opt.get("external_temp") != 1):
            self.k1 = k
            self.tempcnn = None

        # Here is the relaxed continuous KNN
        def indexer(xe_patch, ye_patch):
            return NearestNeighbors(xe_patch, ye_patch, nl_match_window, exclude_self=True)

        self.n3aggregation = res.aggregate(
            indexing=indexer, k=k, patchsize=patchsize, stride=stride, temp_opt=temp_opt)

    def reset_parameters(self):
        m = 0
        while(m < len(self.modules())):

            if isinstance(self.modules()[m], (nn.BatchNorm2d)):
                # Batch Normalization
                ksize = 3
                b_min = 0.025
                n = ksize**2 * self.modules()[m].num_features
                self.modules()[m].weight.data.normal_(0, np.sqrt(2. / (n)))
                self.modules()[m].weight.data[(self.modules()[m].weight.data < 0) & (
                    self.modules()[m].weight.data < b_min)] = b_min
                self.modules()[m].weight.data[(self.modules()[m].weight.data > 0) & (
                    self.modules()[m].weight.data <= b_min)] = b_min
                n += 1
                self.modules()[m].weight.data[(self.modules()[m].weight.data < 0) & (
                    self.modules()[m].weight.data >= -b_min)] = -b_min

                self.modules()[m].weight.data = np.abs(
                    self.modules()[m].weight.data)
                self.modules()[m].bias.data.zero_()
                self.modules()[m].momentum = 0.001
                m += 1

    def forward(self, x):

        # for sending it to the coninuous nearest neighbour network
        x_bck = x
        ytemo = x_bck

        ytemp1 = x

        # embedding network output
        x_embedded = self.embedcnn(x)
        y_embedded = x_embedded
        x_bck = ytemp1
        ltemp = y_embedded
        log_temp = self.tempcnn(x)

        # output from continuous nearest neighbours selection
        y = self.n3aggregation(
            ytemo, x_embedded, y_embedded, log_temp=log_temp)
        return y


# DnCNN Model
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                      kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                          kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                      kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# Final architecture that is DnCNN + N3 block
class neural_nearest_neighbour_network(nn.Module):
    def __init__(self, nplanes_in, out_nplanes, n_planesterm, blocks_n, opt_block, opt_nl, residual=False):

        super(neural_nearest_neighbour_network, self).__init__()
        self.nplanes_in = nplanes_in
        self.out_nplanes = out_nplanes
        self.blocks_n = blocks_n
        self.residual = residual

        nin = nplanes_in
        cnns = []
        nls = []
        i = 0
        while(i < blocks_n-1):
            cnns += [DnCNN(nin, n_planesterm, **opt_block)]
            nl = neural_nearest_neighbourblock(n_planesterm, **opt_nl)
            nin = nl.out_nplanes
            nls += [nl]
            i += 1

        nout = out_nplanes
        cnns += [DnCNN(nin, nout, **opt_block)]

        self.nls = nn.Sequential(*nls)
        self.blocks = nn.Sequential(*cnns)

    def forward(self, x):
        sht = x
        i = 0
        while(i < self.blocks_n-1):
            x = self.blocks_n[i](x)
            x = self.nls[i](x)
            i += 1

        s1 = 0

        x = self.blocks_n[-1](x)

        if self.residual != 1:
            s1 += 1
        elif(self.residual == 1):
            if(self.nplanes_in < self.out_nplanes):
                nsht = self.nplanes_in
            else:
                nsht = self.out_nplanes

            x[:, :nsht, :, :] = x[:, :nsht, :, :] + \
                sht[:, :nsht, :, :]

        print(s1)

        return x
