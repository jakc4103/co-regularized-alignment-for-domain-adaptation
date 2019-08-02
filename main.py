import torch
import torch.nn as nn
import numpy as np
import cv2
import os

from model import build_small_model
from trainer import Trainer
from digit_dataset import get_mnist, get_svhn
from data_loader import get_data_loader, get_lmdb_loader
from config import Configs

import argparse

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, required=True)
    parser.add_argument('--ins_norm', type=bool, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--test_only', action='store_true')
    return parser.parse_args()


def load_weights(net, path, gpu=True):
    if gpu:
        net.load_state_dict(torch.load(path))
    else:
        net.load_state_dict(torch.load(path, map_location='cpu'))

    return net
        

def main():
    args = get_argument()
    config = Configs(args.model_index)
    net = build_small_model(args.ins_norm, False if config.lambda_div == 0 else True)

    if config.mode in [0, -1] and not args.test_only:
        config.dump_to_file(os.path.join(config.save_path, 'exp_config.txt'))
        # random select 1000 sample as validation
        # np.random.seed(1)
        # num_val = 1000
        # rand_idx = np.random.permutation(len(tdata))
        # val_data, val_label = tdata[rand_idx[:num_val]], tlabel[rand_idx[:num_val]]
        # tdata, tlabel = tdata[rand_idx[num_val:]], tlabel[rand_idx[num_val:]]

        #train_data_loader = get_data_loader(sdata, slabel[:, 0], tdata, tlabel[:, 0], batch_size=config.batch_size)
        #val_data_loader = get_data_loader(val_data, val_label[:, 0], val_data, val_label[:, 0], batch_size=64)
        #test_data_loader = get_data_loader(tdata_test, tdata_test_label[:, 0], tdata_test, tdata_test_label[:, 0], shuffle=False, batch_size=64)

        net2 = build_small_model(args.ins_norm, False if config.lambda_div == 0 else True)

        if args.source == 'mnist':
            sname = 'D:/workspace/dataset/digits/MNIST/lmdb'
            tname = 'D:/workspace/dataset/digits/SVHN/lmdb'
        else:
            tname = 'D:/workspace/dataset/digits/MNIST/lmdb'
            sname = 'D:/workspace/dataset/digits/SVHN/lmdb'

        train_data_loader = get_lmdb_loader(sname, tname, 'data', 'label', batch_size=config.batch_size)
        val_data_loader = get_lmdb_loader(sname, tname, 'vdata', 'vlabel', batch_size=config.batch_size)
        test_data_loader = get_lmdb_loader(sname, tname, 'tdata', 'tlabel', batch_size=config.batch_size)

        if config.mode == -1:
            load_weights(net, config.checkpoint)

        trianer = Trainer([net, net2], train_data_loader, val_data_loader, test_data_loader, config)
        trianer.train_all()

    elif config.mode == 1 or args.test_only:
        tdata, tlabel, tdata_test, tdata_test_label = get_svhn()
        #tdata, tlabel, tdata_test, tdata_test_label = get_mnist()
        test_data_loader = get_data_loader(tdata_test, tdata_test_label[:, 0], tdata_test, tdata_test_label[:, 0], shuffle=False, batch_size=32)
        
        load_weights(net, config.checkpoint, config.gpu)
        
        trainer = Trainer([net, net], None, None, test_data_loader, config)
        print('acc', trainer.val_(trainer.nets[0], 0, 0, 'test'))


if __name__=='__main__':
    main()
