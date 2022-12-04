

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
import model_fun as modelfunc


import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
padding = 0
cnn_channels = 0
ksize = 0
cnn_outchannels = 0

cnn_bn = 0
# Parameters
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', default='N3Net', type=str,
                    help='N3Net for Gaussian image denoising')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400',
                    type=str, help='path of train data')
parser.add_argument('--optimizer', default="adam",
                    choices=["adam", "sgd"])  # which optimizer to use
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int,
                    help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
temp_results = []
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


class sum_squared_error(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(
            size_average, reduce, reduction)

    def forward(self, input, target):

        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':

    ninchannels = 1
    noutchannels = 1

    temp_opt = args.nl_temp

    n3block_opt = dict(
        k=args.nl_k,
        patchsize=args.nl_patchsize,
        stride=args.nl_stride,
        temp_opt=temp_opt,
        embedcnn_opt=args.embedcnn)

    dncnn_opt = args.dncnn
    dncnn_opt["residual"] = True
    model = modelfunc.neural_nearest_neighbour_network(ninchannels, noutchannels, args.nfeatures_interm,
                                                       nblocks=args.ndncnn, block_opt=dncnn_opt, nl_opt=n3block_opt, residual=False)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        logger.warning()

        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(
            save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()

    # criterion = nn.MSELoss(reduction = 'sum')
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()

        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()

    logger.warning()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[
                            30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32')/255.0
        logger.warning()
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4,
                             drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        print("***** train metrics *****")

        for n_count, batch_yx in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                logger.warning()
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            loss = criterion(model(batch_y), batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count,
                      xs.size(0)//batch_size, loss.item()/batch_size))
            # temp_results = save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_dncnn'+ext))

        elapsed_time = time.time() - start_time
        print(temp_results)

        print("*** Evaluate ***")
        logger.warning()
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' %
            (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack(
            (epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
