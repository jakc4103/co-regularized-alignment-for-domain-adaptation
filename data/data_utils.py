import os
import numpy as np
from scipy import io
from scipy import misc

import lmdb

def load_mnist_image(path):
    image_list = []
    with open(path, 'rb') as ff:
        ff.seek(0)
        header = np.fromfile(ff, dtype=np.int32, count=4)
        header = header.byteswap()
        #num_img = header[1]
        row = header[2]
        col = header[3]
        while ff:
            img = np.fromfile(ff, dtype=np.uint8, count=row*col)
            if img.shape[0] == 0:
                break
            
            img = img.byteswap().reshape(row, col)
            img = misc.resize(img, (32, 32))
            image_list.append(img)
    return image_list


def load_mnist_label(path):
    label_list = []
    with open(path, 'rb') as ff:
        ff.seek(0)
        header = np.fromfile(ff, dtype=np.int32, count=2)
        header = header.byteswap()

        while ff:
            label = np.fromfile(ff, dtype=np.uint8, count=1)
            if label.shape[0] == 0:
                break
            
            label = label.byteswap()
            label_list.append(label)
    return label_list


def get_mnist(path):
    train_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(path, 'train-labels.idx1-ubyte')
    test_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_label_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    train_img = load_mnist_image(train_path)
    test_img = load_mnist_image(test_path)
    train_label = load_mnist_label(train_label_path)
    test_label = load_mnist_label(test_label_path)

    train_img = np.array(train_img)
    train_img = np.expand_dims(train_img, -1)
    train_img = np.concatenate([train_img, train_img, train_img], -1)

    test_img = np.array(test_img)
    test_img = np.expand_dims(test_img, -1)
    test_img = np.concatenate([test_img, test_img, test_img], -1)

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_img, train_label, test_img, test_label


def get_svhn(path):
    train_path = os.path.join(path, 'train_32x32.mat')
    test_path = os.path.join(path, 'test_32x32.mat')
    train = io.loadmat(train_path)
    train_img = np.transpose(train['X'], (3, 0, 1, 2))
    train_label = train['y']
    train_label[train_label == 10] = 0

    test = io.loadmat(test_path)
    test_img = np.transpose(test['X'], (3, 0, 1, 2))
    test_label = test['y']
    test_label[test_label == 10] = 0

    return train_img, train_label, test_img, test_label


def create_lmdb(mode='train'):
    env = lmdb.open('D:/workspace/dataset/digits/MNIST/lmdb', max_dbs=6, map_size=int(1e9))
    data = env.open_db(("data").encode())
    label = env.open_db(("label").encode())
    vdata = env.open_db(("vdata").encode())
    vlabel = env.open_db(("vlabel").encode())
    tdata = env.open_db(("tdata").encode())
    tlabel = env.open_db(("tlabel").encode())

    dd, ll, tt, ttl = get_mnist()
    #dd, ll, tt, ttl = get_svhn()
    dd = dd.copy(order='C')
    ll = ll.copy(order='C')
    tt = tt.copy(order='C')
    ttl = ttl.copy(order='C')

    # random select 1000 sample as validation
    np.random.seed(1)
    num_val = 1000
    rand_idx = np.random.permutation(len(dd))
    vv, vvl = dd[rand_idx[:num_val]], ll[rand_idx[:num_val]]
    dd, ll = dd[rand_idx[num_val:]], ll[rand_idx[num_val:]]
    
    print(dd[0].shape)
    print(dd[0].dtype)
    #return
    with env.begin(write=True) as txn:
        for idx in range(dd.shape[0]):
            txn.put((str(idx)).encode(), dd[idx], db=data)
            txn.put((str(idx)).encode(), ll[idx], db=label)
        for idx in range(vv.shape[0]):
            txn.put((str(idx)).encode(), vv[idx], db=vdata)
            txn.put((str(idx)).encode(), vvl[idx], db=vlabel)
        for idx in range(tt.shape[0]):
            txn.put((str(idx)).encode(), tt[idx], db=tdata)
            txn.put((str(idx)).encode(), ttl[idx], db=tlabel)


if __name__=='__main__':
    create_lmdb('train')
    