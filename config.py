import os

class Configs:
    def __init__(self, model_index=6):
        self.gpu = True
        self.data_source = 'mnist' #'svhn'
        self.data_target = 'svhn'
        self.total_epoch = 80 # 80 for batch 64
        self.batch_size = 64
        self.lr = 1e-3
        self.lambda_dom = 1e-2 # discriminator
        self.lambda_ent = 1e-2 # conditional entropy
        self.lambda_div = 1e-3 # co-regularized divergence
        self.div_margin = 10 # co-regularized divergence
        self.lambda_agree = 1e-1 # co-regularized agreement
        self.vat_epsilon = 1.0 # vat epsilon
        self.model_index = model_index
        self.ins_norm = True
        self.save_path = 'D:/workspace/DA/model_{}'.format(self.model_index)
        self.source_lmdb = 'D:/workspace/dataset/digits/MNIST/lmdb'
        self.target_lmdb = 'D:/workspace/dataset/digits/SVHN/lmdb'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.mode = 0 # 0 for training, 1 for testing
        self.checkpoint = 'D:/workspace/DA/model_12/model_best_0_0'

    def dump_to_file(self, path):
        with open(path, 'a+') as writer:
            writer.write('mode: {}\nmodel_index:{}\nsource: {}\ntarget: {}\n'.format(self.mode,\
                 self.model_index, self.data_source, self.data_target))
            if self.mode == -1:
                writer.write('checkpoint: {}'.format(self.checkpoint))
            writer.write('save_path: {}\n'.format(self.save_path))

            writer.write('total_epoch: {}\nbatch_size: {}\nlr: {}\n'.format(self.total_epoch, self.batch_size, self.lr))
            writer.write('lambda_dom: {}\n'.format(self.lambda_dom))
            writer.write('lambda_ent: {}\n'.format(self.lambda_ent))
            writer.write('vat_epsilon: {}\n'.format(self.vat_epsilon))
            if self.lambda_div != 0:
                writer.write('lambda_div: {}\n'.format(self.lambda_div))
                writer.write('div_margin: {}\n'.format(self.div_margin))
                writer.write('lambda_agree: {}\n'.format(self.lambda_agree))