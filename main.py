import torch
import torch.nn as nn
import numpy as np
import cv2
import os

from deep_model.model import build_small_model
from trainer import Trainer
from data.data_utils import get_mnist, get_svhn
from data.data_loader import get_data_loader, get_lmdb_loader
from config import Configs

import argparse

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, required=True)
    parser.add_argument('--num_net', type=int, required=True)
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
    net = build_small_model(config.ins_norm, False if config.lambda_div == 0 else True)

    if config.mode in [0, -1] and not args.test_only:
        config.dump_to_file(os.path.join(config.save_path, 'exp_config.txt'))

        train_data_loader = get_lmdb_loader(config.source_lmdb, config.target_lmdb, 'data', 'label', batch_size=config.batch_size)
        val_data_loader = get_lmdb_loader(config.source_lmdb, config.target_lmdb, 'vdata', 'vlabel', batch_size=config.batch_size)
        test_data_loader = get_lmdb_loader(config.source_lmdb, config.target_lmdb, 'tdata', 'tlabel', batch_size=config.batch_size)

        net_list = [net]
        for idx in range(1, args.num_net):
            tmp_net = build_small_model(config.ins_norm, False if config.lambda_div == 0 else True)
            net_list.append(tmp_net)


        trainer = Trainer(net_list, train_data_loader, val_data_loader, test_data_loader, config)

        trainer.train_all()

    elif config.mode == 1 or args.test_only:
        tdata, tlabel, tdata_test, tdata_test_label = get_svhn('D:/workspace/dataset/digits/SVHN/')
        #tdata, tlabel, tdata_test, tdata_test_label = get_mnist('D:/workspace/DA/dataset/MNIST/')
        test_data_loader = get_data_loader(tdata_test, tdata_test_label[:, 0], tdata_test, tdata_test_label[:, 0], shuffle=False, batch_size=32)
        
        load_weights(net, config.checkpoint, config.gpu)
        
        trainer = Trainer([net], None, None, test_data_loader, config)
        print('acc', trainer.val_(trainer.nets[0], 0, 0, 'test'))


if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
