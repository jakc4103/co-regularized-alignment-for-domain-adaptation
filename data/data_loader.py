import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from torchvision import transforms
import lmdb

class Rescale:
    def __call__(self, sample):
        simage, slabel, timage, tlabel = sample['simage'], sample['slabel'], sample['timage'], sample['tlabel']

        simage = ((simage.astype(np.float32) / 255) * 2) - 1
        timage = ((timage.astype(np.float32) / 255) * 2) - 1

        return {'simage': simage, 'slabel': slabel, 'timage': timage, 'tlabel': tlabel}


class ToTensor(object):
    def __call__(self, sample):
        simage, slabel = sample['simage'], sample['slabel']
        timage, tlabel = sample['timage'], sample['tlabel']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        simage = simage.transpose((2, 0, 1))
        timage = timage.transpose((2, 0, 1))
        #label = label.transpose((2, 0, 1))

        return {'simage': simage, 'slabel': slabel, 'timage': timage, 'tlabel': tlabel}


class DigitDataset(Dataset):
    def __init__(self, sdata, slabel, tdata=None, tlabel=None, transform=transforms.Compose([Rescale(), ToTensor()])):
        self.sdata = sdata
        self.slabel = slabel
        self.tdata = tdata
        self.tlabel = tlabel
        self.length = tdata.shape[0]
        self.slength = sdata.shape[0]
        self.tlength = tdata.shape[0]

        self.transform = transform


    def __len__(self):
        return self.length 


    def __getitem__(self, idx):
        if idx >= self.slength:
            sidx = np.random.randint(self.slength)
            simg = self.sdata[sidx]
            slabel = self.slabel[sidx]
        else:
            simg = self.sdata[idx]
            slabel = self.slabel[idx]

        if idx >= self.tlength:
            tidx = np.random.randint(self.tlength)
            timg = self.tdata[tidx]
            tlabel = self.tlabel[tidx]
        else:
            timg = self.tdata[idx]
            tlabel = self.tlabel[idx]

        sample = {'simage': simg, 'slabel': slabel, 'timage': timg, 'tlabel': tlabel}

        if self.transform:
            sample = self.transform(sample)

        return sample


class LMDBDigitDataset(Dataset):
    def __init__(self, sname, tname, dname, lname, transform=transforms.Compose([Rescale(), ToTensor()])):

        self.source_env = lmdb.open(sname, max_dbs=6, map_size=int(1e9), readonly=True)
        self.sdata = self.source_env.open_db(dname.encode())
        self.slabel = self.source_env.open_db(lname.encode())

        self.target_env = lmdb.open(tname, max_dbs=6, map_size=int(1e9), readonly=True)
        self.tdata = self.target_env.open_db(dname.encode())
        self.tlabel = self.target_env.open_db(lname.encode())

        self.source_txn = self.source_env.begin(write=False)
        self.slength = self.source_txn.stat(db=self.sdata)["entries"]

        self.target_txn = self.target_env.begin(write=False)
        self.tlength = self.target_txn.stat(db=self.tdata)["entries"]

        self.length = self.tlength

        self.transform = transform


    def __len__(self):
        return self.length 


    def __getitem__(self, idx):
        if idx >= self.slength:
            sidx = str(int(np.random.randint(self.slength))).encode()
            simg = self.source_txn.get(sidx, db=self.sdata)
            simg = np.frombuffer(simg, 'uint8').reshape(32, 32, 3)
            slabel = self.source_txn.get(sidx, db=self.slabel)
            slabel = np.frombuffer(slabel, 'uint8')[0]#.reshape(1)
        else:
            sidx = str(idx).encode()
            simg = self.source_txn.get(sidx, db=self.sdata)
            simg = np.frombuffer(simg, 'uint8').reshape(32, 32, 3)
            slabel = self.source_txn.get(sidx, db=self.slabel)
            slabel = np.frombuffer(slabel, 'uint8')[0]#.reshape(1)


        if idx >= self.tlength:
            tidx = str(int(np.random.randint(self.tlength))).encode()
            timg = self.target_txn.get(tidx, db=self.tdata)
            timg = np.frombuffer(timg, 'uint8').reshape(32, 32, 3)
            tlabel = self.target_txn.get(tidx, db=self.tlabel)
            tlabel = np.frombuffer(tlabel, 'uint8')[0]#.reshape(1)
        else:
            tidx = str(idx).encode()
            timg = self.target_txn.get(tidx, db=self.tdata)
            timg = np.frombuffer(timg, 'uint8').reshape(32, 32, 3)
            tlabel = self.target_txn.get(tidx, db=self.tlabel)
            tlabel = np.frombuffer(tlabel, 'uint8')[0]#.reshape(1)

        sample = {'simage': simg, 'slabel': slabel, 'timage': timg, 'tlabel': tlabel}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_data_loader(sdata, slabel, tdata, tlabel, batch_size=64, worker=0, shuffle=True):
    dataset = DigitDataset(sdata, slabel, tdata, tlabel)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=worker)
    return dataloader


def get_lmdb_loader(sname, tname, dname, lname, batch_size=64, worker=0, shuffle=True):
    dataset = LMDBDigitDataset(sname, tname, dname, lname)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=worker)

    return dataloader
