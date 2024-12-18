#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
import os
import ast
import pickle
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime
import wandb
import time

from model.dfm import DeepFM
from model.dcn import DCN
from model.dcnv2 import DCNv2
from model.autoint import AutoInt
from model.ple import PLE
from model.mmoe import MMoE
from model.pepnet import PEPNet
from model.star import STAR
from model.aread import AREAD
from model.adl import ADL
from model.hinet import HiNet
from model.adasparse import AdaSparse
from model.mamdr import MAMDR
from dataset.aliccp.preprocess_ali_ccp import reduce_mem


class Run(object):
    def __init__(self, config):
        device = 'cuda:' + str(config.gpu) if config.use_cuda and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = config.model
        self.base_model = config.base_model
        self.epoch = config.epoch
        self.embed_dim = config.embed_dim
        self.seq_maxlen = config.seq_maxlen
        self.domain2encoder_dict = config.domain2encoder_dict
        self.domain2group_list = config.domain2group_org_dict[config.dataset_name][config.group_strategy]
        self.domain2group_dict = {k: self.domain2group_list[k] for k in range(len(self.domain2encoder_dict))}
        self.n_tower = max(self.domain2group_list) + 1
        self.n_domain = None
        self.preprocess_path = config.preprocess_path
        self.domain_filter = config.domain_filter
        self.config = config
        self.preprocess_folder = self.preprocess_path.split('.csv')[0]
        self.dataset_name = config.dataset_name
        if self.dataset_name == 'amazon':
            self.feature_names = ['itemid', 'weekday', 'domain', 'sales_chart', 'sales_rank',
                                  'brand', 'price', 'user_pos_6month_seq', 'user_neg_6month_seq']  # 可能比preprocess中要少
            self.label_name = 'label'
        elif self.dataset_name == 'aliccp':
            categorical_columns = ['userid', '121', '122', '124', '125', '126', '127', '128', '129', 'itemid', 'domain',
                                   '207', '210', '216', '508', '509', '702', '853', '109_14', '110_14', '127_14',
                                   '150_14', '301']
            numerical_columns = []  # ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
            self.feature_names = categorical_columns + numerical_columns
            self.label_name = 'click'
            if len(numerical_columns) == 0:
                print("Warning: no using numerical columns in aliccp dataset")
        elif self.dataset_name == 'cloudtheme':
            self.feature_names = ['userid', 'itemid', 'domain', 'leaf_cate_id', 'cate_level1_id']
            self.label_name = 'click'
        self.multi_hot_flag = None  # 以读入数据时为准，有history, only_id两层选择
        self.one_hot_feature_dims = None
        self.itemid_idx, self.domain_idx = None, None
        self.is_multi_tower = (self.model in ['ple', 'mmoe', 'pepnet', 'epnet', 'star', 'adl', 'hinet'])
        self.is_concat_group = (self.model in ['star', 'adl', 'hinet'])
        self.selected_domain, self.train_loss_domain_group = None, None
        self.train_valid, self.valid_test = None, None
        self.is_aug_data = ('aread' in self.model)
        self.domain_cnt_weight = None
        self.train_domain_batch_seq, self.valid_domain_batch_seq, self.test_domain_batch_seq = [], [], []
        self.train_data_loader, self.valid_data_loader, self.test_data_loader = None, None, None
        self.train_data_generator, self.valid_data_generator, self.test_data_generator = None, None, None
        self.aug_train_data_loader, self.aug_train_data_generator = None, None  # for train data augmentation
        self.regroup_times = 0
        self.random_modify_sigma = self.config.random_modify_sigma
        self.candidate_mask_num = self.config.candidate_mask_num * 1.
        self.init_active_percent = self.config.init_active_percent

        # find the latest model
        # all_models = [f for f in os.listdir(self.config.save_path) if f.startswith(f'{self.model}') and f.endswith('.pth.tar')]
        # if all_models:
        #     latest_model_inx = max([int(x.split('_')[1].split('.')[0]) for x in all_models])
        # else:
        #     latest_model_inx = 0
        self.latest_model_inx = np.random.randint(50)
        self.save_model_path = os.path.join(self.config.save_path, f'{self.model}_{self.latest_model_inx+1}.pth.tar')
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
            print(f'create save_path folder: {self.config.save_path}')
        print('save_model_path: ', self.save_model_path)

        # for early stop
        self.num_trials = config.early_stop
        self.trial_counter = 0
        self.best_loss, self.best_mean_loss = np.inf, np.inf
        self.best_auc, self.best_mean_auc = 0, 0

    def seq_extractor(self, seq, maxlen):
        seq = ast.literal_eval(seq)
        padding_value = self.config.itemid_all
        if len(seq) >= maxlen:
            return np.array(seq[-maxlen:])
        else:
            return np.array(seq + [padding_value] * (maxlen - len(seq)))

    def read_split_data(self, path, aug_path, history=True, only_id=False):
        print('data path: ', path)
        if only_id:
            x_cols = ['userid', 'itemid', 'domain']
            self.feature_names = x_cols
            print('only id features and domain')
        else:
            x_cols = [fea for fea in self.feature_names if 'seq' not in fea]
            if history:
                x_cols.extend([fea for fea in self.feature_names if 'seq' in fea])
            else:
                print('no history, no multi-hot features')

            self.feature_names = x_cols

        print('feature_names: ', self.feature_names)
        y_col = [self.label_name]
        return_cols = x_cols + y_col
        if self.dataset_name == 'amazon':
            # 使用timestamp作为划分数据集的依据
            split_col = 'timestamp'
        elif self.dataset_name in ['aliccp', 'cloudtheme']:
            # preprocess_ali_ccp.py中已经将数据集划分好了，预处理时打了train_tag标签
            split_col = 'train_tag'  # train_tag: 0-train, 1-val, 2-test
        cols = x_cols + y_col + [split_col]

        data = reduce_mem(pd.read_csv(path, usecols=cols))
        # data.sort_values(by=['timestamp', 'domain'], inplace=True)
        if self.dataset_name == 'amazon':
            self.train_valid, self.valid_test = data[split_col].quantile(0.9), data[split_col].quantile(0.95)
        else:
            self.train_valid, self.valid_test = 1, 2

        if self.domain_filter is not None:
            self.domain_filter = ast.literal_eval(self.domain_filter)
            print(f'filter domain: {self.domain_filter}')
            data = data.loc[data['domain'].isin(self.domain_filter)].copy()

        self.itemid_idx = x_cols.index('itemid')
        self.domain_idx = x_cols.index('domain')
        one_hot_cols = [col for col in x_cols if 'seq' not in col]
        self.one_hot_feature_dims = np.max(data[one_hot_cols], axis=0) + 1  # length of one-hot
        if self.dataset_name == 'amazon':
            # 因为有multi-hot，所以itemid.max可能小于itemid_all
            self.one_hot_feature_dims[self.itemid_idx] = self.config.itemid_all
        self.multi_hot_flag = [False] * len(one_hot_cols) + [True] * (len(x_cols) - len(one_hot_cols)) * self.seq_maxlen
        self.n_domain = data['domain'].nunique()

        if self.is_multi_tower:
            print(f'group num in multi-tower: {self.n_tower}')
            print('domain2group_list: ', self.domain2group_list)

        wandb.log({'preprocess_folder': self.preprocess_folder, 'save_model_path': self.save_model_path,
                   'feature_names': self.feature_names, 'multi_hot_flag': self.multi_hot_flag})

        if self.domain_filter is None:  # (self.config.is_set_seed == 0) and (self.domain_filter is None):
            # 高效跑程序模式（无随机种子）且不需要筛domain->不做数据统计，直接读取预处理好的数据
            del data
            return return_cols, (None, None, None, None)

        if self.config.is_evaluate_multi_domain:
            damain_counts = data['domain'].value_counts()
            domain_positive = data.groupby(by='domain')[self.label_name].sum()
            domain_statistics = pd.concat([damain_counts, domain_positive], axis=1)
            print('counts of each domain in total data:')
            print(domain_statistics)

        train_data = data[data[split_col] < self.train_valid]
        valid_data = data[(data[split_col] >= self.train_valid) & (data[split_col] < self.valid_test)]
        test_data = data[data[split_col] >= self.valid_test]

        data_len = [train_data.shape[0], valid_data.shape[0], test_data.shape[0]]
        print('before multi-hot features flatten, including input features, timestamps and labels')
        print(f'train_data:{train_data.shape}, valid_data:{valid_data.shape}, test_data:{test_data.shape}')
        print(f'train:valid:test = '
              f'{data_len[0]/sum(data_len):.2f}:{data_len[1]/sum(data_len):.2f}:{data_len[2]/sum(data_len):.2f}')
        if self.dataset_name == 'amazon':
            print(f'train time: {datetime.fromtimestamp(train_data["timestamp"].min()).strftime("%Y-%m-%d")} '
                  f'to {datetime.fromtimestamp(train_data["timestamp"].max()).strftime("%Y-%m-%d")}')
            print(f'valid time: {datetime.fromtimestamp(valid_data["timestamp"].min()).strftime("%Y-%m-%d")} '
                  f'to {datetime.fromtimestamp(valid_data["timestamp"].max()).strftime("%Y-%m-%d")}')
            print(f'test  time: {datetime.fromtimestamp(test_data["timestamp"].min()).strftime("%Y-%m-%d")} '
                  f'to {datetime.fromtimestamp(test_data["timestamp"].max()).strftime("%Y-%m-%d")}')

        # count overlap
        # print('Calculate the user overlap between train_data, valid_data and test_data')
        # train_user_ids = set(train_data['userid'].unique())
        # valid_user_ids = set(valid_data['userid'].unique())
        # test_user_ids = set(test_data['userid'].unique())
        # train_valid_user_inter = len(train_user_ids.intersection(valid_user_ids))
        # train_test_user_inter = len(train_user_ids.intersection(test_user_ids))
        # print(f'{train_valid_user_inter}/{len(valid_user_ids)} ({(train_valid_user_inter/len(valid_user_ids)):.2f}) '
        #       f'users in valid_data exists in train_data')
        # print(f'{train_test_user_inter}/{len(test_user_ids)} ({(train_test_user_inter/len(test_user_ids)):.2f}) '
        #       f'users in test_data exists in train_data')

        print('Calculate the item overlap between train_data, valid_data and test_data')
        train_item_ids = set(train_data['itemid'].unique())
        valid_item_ids = set(valid_data['itemid'].unique())
        test_item_ids = set(test_data['itemid'].unique())
        train_valid_item_inter = len(train_item_ids.intersection(valid_item_ids))
        train_test_item_inter = len(train_item_ids.intersection(test_item_ids))
        print(f'{train_valid_item_inter}/{len(valid_item_ids)} ({(train_valid_item_inter/len(valid_item_ids)):.2f}) '
              f'items in valid_data exists in train_data')
        print(f'{train_test_item_inter}/{len(test_item_ids)} ({(train_test_item_inter/len(test_item_ids)):.2f}) '
              f'items in test_data exists in train_data')

        if self.is_aug_data:
            aug_data = pd.read_csv(aug_path, usecols=cols)
            aug_train_data = aug_data[aug_data[split_col] < self.train_valid]
            if self.config.is_evaluate_multi_domain:
                damain_counts = train_data['domain'].value_counts()
                aug_domain_counts = aug_train_data['domain'].value_counts()
                concat_domain_counts = pd.concat([damain_counts, aug_domain_counts], axis=1)
                print('domain compare between original data & augmented data')
                print(concat_domain_counts)
        else:
            aug_train_data = pd.DataFrame(columns=cols)

        return return_cols, (train_data[return_cols],
                             valid_data[return_cols],
                             test_data[return_cols],
                             aug_train_data[return_cols])

    def save_tensor_from_data(self, data, cols, mode):
        x_cols = [col for col in cols if col != self.label_name]
        y_col = [self.label_name]
        seq_cols = [col for col in cols if 'seq' in col]
        print('save_tensor_from_data: x_cols:', x_cols)
        print('save_tensor_from_data: seq_cols:', seq_cols)
        if len(seq_cols) > 0:
            seq_tensors = None

            from keras.preprocessing.sequence import pad_sequences
            for col in seq_cols:
                data[col] = data[col].apply(self.seq_extractor, maxlen=self.seq_maxlen)
                seq = pad_sequences(data[col], maxlen=self.seq_maxlen, padding='post', value=self.config.itemid_all)
                if seq_tensors is None:
                    seq_tensors = torch.tensor(seq, dtype=torch.int)
                else:
                    seq_tensors = torch.cat([seq_tensors, torch.tensor(seq, dtype=torch.int)], dim=1)
            id_cols = [col for col in x_cols if 'seq' not in col]
            id_tensors = torch.tensor(data[id_cols].values, dtype=torch.int)
            X = torch.cat([id_tensors, seq_tensors], dim=1)
        else:
            X = torch.tensor(data[x_cols].values, dtype=torch.int)
        y = torch.tensor(data[y_col].values, dtype=torch.short)
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)
        torch.save(X, os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth'))
        torch.save(y, os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth'))

        return X, y

    def convert2data_loader(self, data, cols, mode):
        cols = cols if data is None else data.columns
        x_cols = [col for col in cols if col != self.label_name]
        seq_cols = [col for col in cols if 'seq' in col]
        y_col = [self.label_name]

        if os.path.exists(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')):
            X = torch.load(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')).to(torch.int)
            y = torch.load(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')).to(torch.short)
        elif os.path.exists(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')):
            X = torch.load(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')).to(torch.int)
            y = torch.tensor(data[y_col].values, dtype=torch.short)
            torch.save(y, os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth'))
        else:
            X, y = self.save_tensor_from_data(data, cols, mode)

        assert X.shape[1] == (len(x_cols)+len(seq_cols)*(self.seq_maxlen-1)), \
            f'X.shape[1] != len(x_cols): {X.shape[1]} != {len(x_cols)+len(seq_cols)*(self.seq_maxlen-1)}'

        if self.domain_filter is not None:
            mask = torch.isin(X[:, self.domain_idx], torch.tensor(self.domain_filter))
            X = X[mask]
            y = y[mask]
        if self.is_multi_tower:
            group = pd.Series(X[:, self.domain_idx]).map(self.domain2group_dict)
            group = torch.tensor(group.values, dtype=torch.int64).view(-1, 1).to(self.device)

        if self.config.is_evaluate_multi_domain and mode == 'train':
            domain_val = X[:, self.domain_idx]
            domain_cnt = domain_val.bincount()
            self.domain_cnt_weight = np.array([domain_cnt[i]/X.shape[0] for i in range(len(domain_cnt))])

        print(f'{mode}_data: {X.shape}, {y.shape}', end='')

        X, y = X.to(self.device), y.to(self.device)
        if self.is_multi_tower:
            dataset = TensorDataset(X, y, group)
        else:
            dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, self.config.bs, shuffle=True)

        return data_loader

    def convert2domain_data_loader(self, data, cols, mode):
        cols = cols if data is None else data.columns
        x_cols = [col for col in cols if col != 'label']
        y_col = ['label']

        if os.path.exists(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')):
            X = torch.load(os.path.join(self.preprocess_folder, f'{mode}_data_loader.pth')).to(torch.int)
            y = torch.load(os.path.join(self.preprocess_folder, f'{mode}_label_loader.pth')).to(torch.short)
        else:
            X, y = self.save_tensor_from_data(data, cols, mode)

        if self.domain_filter is not None:
            mask = torch.isin(X[:, self.domain_idx], torch.tensor(self.domain_filter))
            X = X[mask]
            y = y[mask]

        domain_data_loader = []
        domain_list = self.domain_filter if self.domain_filter is not None else range(self.n_domain)
        for d in domain_list:
            if self.domain_filter is not None and d not in self.domain_filter:
                continue
            mask = (X[:, self.domain_idx] == d)
            domain_X = X[mask]
            domain_y = y[mask]
            domain_X, domain_y = domain_X.to(self.device), domain_y.to(self.device)
            domain_data_loader.append(DataLoader(TensorDataset(domain_X, domain_y), self.config.bs, shuffle=True))
            if mode == 'train':
                self.train_domain_batch_seq.extend([d]*np.ceil(domain_X.shape[0]*1./self.config.bs).astype(int))
            elif mode == 'valid':
                self.valid_domain_batch_seq.extend([d]*np.ceil(domain_X.shape[0]*1./self.config.bs).astype(int))
            elif mode == 'test':
                self.test_domain_batch_seq.extend([d]*np.ceil(domain_X.shape[0]*1./self.config.bs).astype(int))
        if mode == 'train':
            domain_val = X[:, self.domain_idx]
            domain_cnt = domain_val.bincount()
            print(mode, 'domain data cnt: ', domain_cnt)
            self.domain_cnt_weight = np.array([domain_cnt[i]/X.shape[0] for i in range(len(domain_cnt))])
            np.random.shuffle(self.train_domain_batch_seq)  # shuffle train_domain_batch_seq
        elif mode == 'valid':
            np.random.shuffle(self.valid_domain_batch_seq)
        elif mode == 'test':
            np.random.shuffle(self.test_domain_batch_seq)

        return domain_data_loader

    def get_data(self):
        print('========Reading data========')
        if self.is_aug_data:
            cols, data = self.read_split_data(self.preprocess_path, self.config.preprocess_aug_path)
        else:
            cols, data = self.read_split_data(self.preprocess_path, None)
        print('after multi-hot features flatten')
        if 'aread' in self.model and ('wo' not in self.model):
            self.train_data_loader = self.convert2domain_data_loader(data[0], cols, mode='train')
            self.valid_data_loader = self.convert2domain_data_loader(data[1], cols, mode='valid')
            self.test_data_loader = self.convert2domain_data_loader(data[2], cols, mode='test')
        else:
            self.train_data_loader = self.convert2data_loader(data[0], cols, mode='train')
            self.valid_data_loader = self.convert2data_loader(data[1], cols, mode='valid')
            self.test_data_loader = self.convert2data_loader(data[2], cols, mode='test')
        if self.is_aug_data:
            self.aug_train_data_loader = self.convert2domain_data_loader(data[3], cols,
                                           mode='aug{0:.1f}_train'.format(self.config.aug_ratio))

        print('\n========Finish reading data========')
        return self.train_data_loader, self.valid_data_loader, self.test_data_loader

    def get_model(self):
        multi_hot_dict = {'multi_hot_flag': self.multi_hot_flag,
                          'itemid_idx': self.itemid_idx,
                          'seq_maxlen': self.seq_maxlen,
                          'method': 'mean'}
        if self.model == 'deepfm':
            model = DeepFM(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict, mlp_dims=(256, 128))
        elif self.model == 'dcn':
            model = DCN(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                        n_cross_layers=3, mlp_dims=self.config.mlp_dims)
        elif self.model == 'dcnv2':
            model = DCNv2(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                          n_cross_layers=3, mlp_dims=self.config.mlp_dims)
        elif self.model == 'autoint':
            model = AutoInt(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                            atten_embed_dim=64, mlp_dims=self.config.mlp_dims)
        elif self.model == 'ple':
            model = PLE(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                        n_tower=self.n_tower,
                        n_expert_specific=self.config.ple_n_expert_specific,
                        n_expert_shared=self.config.ple_n_expert_shared,
                        expert_dims=self.config.ple_expert_dims,
                        tower_dims=self.config.ple_tower_dims, config=self.config)
        elif self.model == 'mmoe':
            model = MMoE(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                         n_tower=self.n_tower, n_expert=self.config.mmoe_n_expert,
                         expert_dims=self.config.mmoe_expert_dims,
                         tower_dims=self.config.mmoe_tower_dims, config=self.config)
        elif self.model == 'pepnet':
            model = PEPNet(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                           n_tower=self.n_tower, tower_dims=self.config.tower_dims, gate_hidden_dim=64,
                           domain_idx=self.domain_idx, use_ppnet=True,
                           config=self.config)
        elif self.model == 'epnet':
            model = PEPNet(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                           n_tower=self.n_tower, tower_dims=self.config.tower_dims, gate_hidden_dim=64,
                           domain_idx=self.domain_idx, use_ppnet=False, config=self.config)
        elif self.model == 'epnet-single':
            model = PEPNet(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                           n_tower=1, tower_dims=self.config.tower_dims, gate_hidden_dim=64,
                           domain_idx=self.domain_idx, use_ppnet=False, config=self.config)
        elif self.model == 'star':
            model = STAR(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                         n_tower=self.n_tower, tower_dims=self.config.tower_dims,
                         domain_idx=self.domain_idx, config=self.config)
        elif self.model == 'adl':
            model = ADL(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict, n_tower=self.n_tower,
                        tower_dims=self.config.tower_dims, domain_idx=self.domain_idx, dlm_iters=self.config.dlm_iters,
                        device=self.device, config=self.config)
        elif self.model == 'hinet':
            model = HiNet(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                          n_tower=self.n_tower, sei_dims=self.config.sei_dims, tower_dims=self.config.tower_dims,
                          domain_idx=self.domain_idx, device=self.device, config=self.config)
        elif self.model == 'adasparse':
            model = AdaSparse(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                              hidden_dims=self.config.mlp_dims, domain_idx=self.domain_idx,
                              config=self.config)
        elif self.model == 'mamdr':
            model = MAMDR(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict, mlp_dims=(256, 128))

        elif self.model == 'aread' or self.model == 'aread_womask':
            n_tower = []
            for l in range(len(self.config.aread_tower_dims)):
                n_tower.append(self.n_tower*(2**l))
            model = AREAD(self.one_hot_feature_dims, self.embed_dim, multi_hot_dict,
                          n_tower=tuple(n_tower),
                          n_domain=self.n_domain,
                          base_model=self.base_model,
                          expert_dims=self.config.mlp_dims,
                          tower_dims=self.config.aread_tower_dims,
                          domain_idx=self.domain_idx,
                          device=self.device,
                          config=self.config
                          )
            # with open(os.path.join(self.config.data_path, 'hierarchical_cluster', 'cluster_z.pkl'), 'rb') as f:
            #     cluster_z_dict = pickle.load(f)
            # model.create_domain_mask(cluster_z_dict['cluster_z_'+str(self.config.hier_cluster_gini_weight)])
            model.reset_for_mask_update()
        else:
            raise ValueError('Unknown model: ' + self.model)
        return model.to(self.device)

    def is_continuable(self, model, result_dict, epoch_i, optimizer):
        # if result_dict['total_auc'] > self.best_auc:
        if result_dict['mean_auc'] > self.best_mean_auc:  # use mean_auc to early stop
            self.trial_counter = 0
            self.best_auc = result_dict['total_auc']
            self.best_loss = result_dict['total_loss']
            save_dict = {'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'best_auc': self.best_auc,
                         'best_result': result_dict, 'preprocess_path': self.preprocess_path,
                         'optimizer': optimizer.state_dict()}
            if result_dict.get('mean_auc') is not None:
                self.best_mean_auc = result_dict['mean_auc']
                self.best_mean_loss = result_dict['mean_loss']
                save_dict['best_mean_auc'] = self.best_mean_auc
                save_dict['best_mean_loss'] = self.best_mean_loss
            if 'aread' in self.model:
                save_dict['domain_mask'] = model.domain_mask

            torch.save(save_dict, self.save_model_path)
            print(f'current best epoch: {epoch_i - self.trial_counter + 1}, '
                  f'auc: {self.best_auc:.4f}, loss: {self.best_loss:.4f}')
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

    def train(self, data_loader, model, criterion, optimizer, epoch_i):
        print('Training Epoch {}:'.format(epoch_i + 1))
        model.train()
        loss_sum = 0
        log_interval = 204800//self.config.bs
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=30)
        for i, batch in enumerate(tk0):
            if self.is_concat_group:
                X, y, group = batch
                pred, y = model(X, group, targets=y)
                loss = criterion(pred.squeeze(), y.squeeze().float())
            elif self.is_multi_tower:
                X, y, group = batch
                pred = model(X)
                loss = criterion(pred.gather(1, group).squeeze(1), y.squeeze().float())
            else:
                X, y = batch
                pred = model(X)
                loss = criterion(pred.squeeze(), y.squeeze().float())
            loss = loss + model.get_regularization_loss(device=self.device)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                wandb.log({'train_loss': (loss_sum / log_interval)})
                loss_sum = 0

    def regroup_all_domain(self, regroup_mode=('towerfirst',)):
        domain2group = [-1]*self.n_domain
        if 'served' in regroup_mode:
            # 保留已经选好的domain的原有分组
            for g in range(self.n_tower):
                domain2group[self.selected_domain[g]] = g
        if 'besttower' in regroup_mode:
            # 选择loss结果最好的Tower
            for d in range(self.n_domain):
                if domain2group[d] < 0:
                    domain2group[d] = np.argmin(self.train_loss_domain_group[:, d])
        elif 'towerfirst' in regroup_mode:
            # 先确保每个塔最好的domain在该group，而后再为每个domain选择自己最好的tower
            for g in range(self.n_tower):
                best_d4g = np.argmin(self.train_loss_domain_group[g, :])
                if domain2group[best_d4g] < 0:
                    domain2group[best_d4g] = g
            for d in range(self.n_domain):
                if domain2group[d] < 0:
                    domain2group[d] = np.argmin(self.train_loss_domain_group[:, d])
        domain2group = np.array(domain2group)
        # check all domains are grouped
        assert np.all(domain2group >= 0)
        return domain2group

    def get_losses_tower_domain(self, pred, y, domains, criterion):
        targets = y.squeeze().float()
        domains = domains.squeeze()
        tower_domain_loss = np.zeros((self.n_tower, self.n_domain))
        for g in range(self.n_tower):
            predicts = pred[:, g].squeeze()
            for d in range(self.n_domain):
                mask = (domains == d).squeeze()
                tower_domain_loss[g][d] = criterion(predicts[mask], targets[mask]).item()
        return tower_domain_loss

    def get_domain_data(self, d, mode='train'):
        if mode == 'train':
            try:
                return next(self.train_data_generator[d])
            except StopIteration:
                self.train_data_generator[d] = iter(self.train_data_loader[d])
                return next(self.train_data_generator[d])
        elif mode == 'valid':
            try:
                return next(self.valid_data_generator[d])
            except StopIteration:
                self.valid_data_generator[d] = iter(self.valid_data_loader[d])
                return next(self.valid_data_generator[d])
        elif mode == 'test':
            try:
                return next(self.test_data_generator[d])
            except StopIteration:
                self.test_data_generator[d] = iter(self.test_data_loader[d])
                return next(self.test_data_generator[d])
        elif mode == 'aug_train':
            try:
                return next(self.aug_train_data_generator[d])
            except StopIteration:
                self.aug_train_data_generator[d] = iter(self.aug_train_data_loader[d])
                return next(self.aug_train_data_generator[d])


    def train_aread(self, model, criterion, optimizer, epoch_i):
        print('Training Epoch {}:'.format(epoch_i + 1))
        model.train()
        loss_sum = 0
        log_interval = 204800//self.config.bs

        warm_up_interval = (self.config.warm_up_interval*1024)//self.config.bs  # 默认100
        regroup_interval = (self.config.regroup_interval*1024)//self.config.bs  # 大约是1/5个epoch重新选一次mask
        print('warm_up_interval: ', warm_up_interval, '; regroup_interval: ', regroup_interval)

        if epoch_i == 0:
            print('========Warm up========')  # warm up的过程中考虑领域均衡性，因此使用aug数据
            tk0 = tqdm.tqdm(range(warm_up_interval), smoothing=0, mininterval=20)
            domain_list = list(range(self.n_domain))
            for i in tk0:
                if len(domain_list) == 0:
                    domain_list = list(range(self.n_domain))
                d = domain_list.pop()
                domain_train_X, domain_train_y = self.get_domain_data(d)
                pred = model(domain_train_X, mode='wo_mask', domain_i=d, memory_gate_value=True)
                loss = criterion(pred.squeeze(), domain_train_y.squeeze().float())
                loss = loss + model.get_regularization_loss(device=self.device)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                if (i + 1) % log_interval == 0:
                    tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                    wandb.log({'train_loss': (loss_sum / log_interval)})
                    loss_sum = 0

        tk0 = tqdm.tqdm(self.train_domain_batch_seq, smoothing=0, mininterval=30)
        for i, d in enumerate(tk0):
            train_X, train_y = self.get_domain_data(d)
            if (epoch_i == 0 and i == 0) or ((i+1) % regroup_interval == 0):
                # 刚刚warm up完，或者训练一定轮数后，需要重新选mask
                model.save_model_state()
                self.random_modify_sigma = self.random_modify_sigma * 0.99  # 逐渐减小随机修改的幅度
                self.init_active_percent = max(0.1, self.init_active_percent * 0.95)
                self.candidate_mask_num = self.candidate_mask_num * 0.99
                current_candidate_mask_num = max(1, int(self.candidate_mask_num))
                print(f'epoch: {epoch_i}, regroup_times: {self.regroup_times}, '
                      f'random_modify_sigma: {self.random_modify_sigma}, '
                      f'init_active_percent: {self.init_active_percent}, '
                      f'candidate_mask_num: {current_candidate_mask_num}')
                self.regroup_times += 1
                tk1 = tqdm.tqdm(range(self.n_domain), desc='Update mask for domain', leave=False, mininterval=60)
                for d in tk1:
                    # tk1.set_description(f"Update mask for domain {d}")
                    for z in range(current_candidate_mask_num):
                        tmp_mask = model.generate_mask(generate_mode='mask_max_gate',
                                                       d=d, init_active_percent=self.init_active_percent,
                                                       random_modify_sigma=self.random_modify_sigma)
                        model.load_model_state()
                        optimizer_fast = torch.optim.Adam(model.parameters(), lr=self.config.update_lr,
                                                          betas=(0.9, 0.99), eps=1e-8, weight_decay=self.config.wd)
                        for s in range(self.config.regroup_update_step):
                            domain_train_X, domain_train_y = self.get_domain_data(d, mode='aug_train')  # 用aug数据
                            # pred = model(domain_train_X, mode='domain_with_mask',
                            #              current_mask=tmp_mask, tmp_memory_gate_value=True)
                            # loss = criterion(pred.squeeze(), domain_train_y.squeeze().float())
                            # loss = loss + model.get_regularization_loss(device=self.device)
                            preds = model(domain_train_X, mode='domain_mask_bagging',
                                          current_mask=tmp_mask, tmp_memory_gate_value=True)
                            domain_train_y = domain_train_y.squeeze().float()
                            losses = [criterion(pred, domain_train_y) for pred in preds.unbind(dim=0)]
                            loss = sum(losses) / preds.shape[0] + model.get_regularization_loss(device=self.device)
                            model.zero_grad()
                            loss.backward()
                            optimizer_fast.step()
                            tmp_mask = model.prun_single_mask(d, tmp_mask, prun_ratio=0.05)  # 逐步剪枝
                        model.candidate_domain_mask[d].append(tmp_mask)

                        with torch.no_grad():
                            for s in range(self.config.regroup_eval_step):
                                domain_train_X, domain_train_y = self.get_domain_data(d)  # 用原数据
                                pred = model(domain_train_X, mode='domain_with_mask', current_mask=tmp_mask)
                                loss = criterion(pred.squeeze(), domain_train_y.squeeze().float())
                                loss = loss + model.get_regularization_loss(device=self.device)
                                model.add_eval_loss(loss.mean().item(), d=d, mask_z=z)
                    # tk1.refresh()  # 刷新显示
                model.update_all_mask(regroup_times=self.regroup_times)
                model.reset_for_mask_update()
                model.load_model_state()

                preds = model(train_X, mode='domain_mask_bagging', domain_i=d)
            else:
                if (i+1)//regroup_interval-(i+1+warm_up_interval)//regroup_interval > 0:
                    preds = model(train_X, mode='domain_mask_bagging', memory_gate_value=True, domain_i=d)
                else:
                    preds = model(train_X, mode='domain_mask_bagging', domain_i=d)

            # loss = criterion(pred.squeeze(), train_y.squeeze().float())
            # loss = loss + model.get_regularization_loss(device=self.device)
            train_y = train_y.squeeze().float()
            try:
                losses = [criterion(pred, train_y) for pred in preds.unbind(dim=0)]
            except:
                print(preds.shape(), domain_train_y.shape())
            loss = sum(losses) / preds.shape[0] + model.get_regularization_loss(device=self.device)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                wandb.log({'train_loss': (loss_sum / log_interval)})
                loss_sum = 0

    def train_aread_final(self, model, criterion, optimizer, train_domain_batch_seq, epoch_i):
        print('Training Final Gate Epoch {}:'.format(epoch_i + 1))
        model.train()
        loss_sum = 0
        log_interval = 102400//self.config.bs

        for i in range(self.n_domain):
            if i not in train_domain_batch_seq:
                train_domain_batch_seq.append(i)  # 保证每个domain都被训练到
        tk0 = tqdm.tqdm(train_domain_batch_seq, smoothing=0, mininterval=30)
        for i, d in enumerate(tk0):
            train_X, train_y = self.get_domain_data(d)
            pred = model(train_X, mode='domain_mask_final', domain_i=d)
            loss = criterion(pred.squeeze(), train_y.squeeze().float())
            loss = loss + model.get_regularization_loss(device=self.device)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                wandb.log({'train_loss': (loss_sum / log_interval)})
                loss_sum = 0

    def test(self, data_loader, model, mode='valid', aread_final=False):
        print('Evaluating:')
        model.eval()
        targets, predicts, domains = [], [], []

        with torch.no_grad():
            if 'aread' in self.model and ('wo' not in self.model):
                for d in tqdm.tqdm(self.valid_domain_batch_seq if mode == 'valid' else self.test_domain_batch_seq,
                                   smoothing=0, mininterval=60):
                    X, y = self.get_domain_data(d, mode=mode)
                    pred = model(X, mode='domain_with_mask', domain_i=d) if not aread_final else \
                        model(X, mode='domain_mask_final', domain_i=d)

                    targets.append(y.squeeze().cpu().numpy())
                    predicts.append(pred.squeeze().cpu().numpy())
                    domains.append(X[:, self.domain_idx].cpu().numpy())
            else:
                for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=60):
                    if self.is_concat_group:
                        X, y, group = batch
                        pred, y = model(X, group, targets=y)
                    elif self.is_multi_tower:
                        X, y, group = batch
                        pred = model(X).gather(1, group)
                    else:
                        X, y = batch
                        pred = model(X)
                    targets.append(y.squeeze().cpu().numpy())
                    predicts.append(pred.squeeze().cpu().numpy())
                    domains.append(X[:, self.domain_idx].cpu().numpy())

        targets = np.concatenate(targets)
        predicts = np.concatenate(predicts)
        domains = np.concatenate(domains)

        # if self.config.run_cnt == 99:
        #     df = pd.DataFrame({
        #         'Target': targets,
        #         'Predict': predicts,
        #         'Domain': domains
        #     })
        #     df.to_csv("result/aliccp/single_result4split.csv", index=False)


        result_dict = dict()
        result_dict['total_auc'] = roc_auc_score(targets, predicts)
        result_dict['total_loss'] = log_loss(targets, predicts)
        # print(type(result_dict['total_auc']), type(result_dict['total_loss']))
        if self.config.is_evaluate_multi_domain:
            result_dict.update(self.evaluate_multi_domain(targets, predicts, domains))

        return result_dict

    def get_domain_loss(self, data_loader, model, mode='valid'):
        print('Evaluating:')
        model.eval()
        targets, predicts, domains, losses = [], [], [], []
        import torch.nn as nn
        bce_loss = nn.BCELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=60):
                if self.is_multi_tower:
                    X, y, group = batch
                    pred = model(X).gather(1, group).squeeze(1)
                else:
                    X, y = batch
                    pred = model(X)
                losses.append(bce_loss(pred.squeeze(), y.squeeze().float()).detach().cpu().numpy())
                domains.append(X[:, self.domain_idx].cpu().numpy())

        losses = np.concatenate(losses)
        domains = np.concatenate(domains)

        return domains, losses

    def evaluate_multi_domain(self, targets, predicts, domains, return_type='dict'):
        df = pd.DataFrame({'targets': targets, 'predicts': predicts, 'domains': domains})
        if return_type == 'dict':
            domain_auc, domain_loss = dict(), dict()
        else:
            domain_auc, domain_loss = np.zeros(self.n_domain), np.zeros(self.n_domain)
        mean_auc, mean_loss = 0, 0

        for domain, group in df.groupby('domains'):
            try:
                auc = roc_auc_score(group['targets'], group['predicts'])
                loss = log_loss(group['targets'], group['predicts'])
            except ValueError:
                # 处理无法计算 AUC 或 Loss 的情况（例如，某个类别的标签只有一种）
                auc, loss = np.nan, np.nan

            domain_auc[domain], domain_loss[domain] = auc, loss
            mean_auc += self.domain_cnt_weight[domain] * auc
            mean_loss += self.domain_cnt_weight[domain] * loss

        return dict({'domain_auc': domain_auc, 'domain_loss': domain_loss,
                     'mean_auc': mean_auc, 'mean_loss': mean_loss})

    def get_kl_loss(self):
        train_data_loader, valid_data_loader, test_data_loader = self.get_data()
        model = self.get_model()
        self.save_model_path = os.path.join(self.config.save_path, f'{self.model}_16.pth.tar')
        print('loading model from', self.save_model_path)
        checkpoint = torch.load(self.save_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        domains, losses = self.get_domain_loss(test_data_loader, model, mode='test')
        np.save(os.path.join(self.config.save_path, f'{self.model}_domains.npy'), domains)
        np.save(os.path.join(self.config.save_path, f'{self.model}_losses.npy'), losses)

    def main(self):
        train_data_loader, valid_data_loader, test_data_loader = self.get_data()
        if 'aread' in self.model and ('wo' not in self.model):
            self.train_data_generator = [iter(self.train_data_loader[d]) for d in range(self.n_domain)]
            self.valid_data_generator = [iter(self.valid_data_loader[d]) for d in range(self.n_domain)]
            self.test_data_generator = [iter(self.test_data_loader[d]) for d in range(self.n_domain)]
        if self.is_aug_data:
            self.aug_train_data_generator = [iter(self.aug_train_data_loader[d]) for d in range(self.n_domain)]
        model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     betas=(0.9, 0.99), eps=1e-8, weight_decay=self.config.wd)

        criterion = torch.nn.BCELoss(reduction='mean')

        if self.config.is_increment:
            print('loading model for increment learning...')
            checkpoint = torch.load(os.path.join(self.config.save_path,
                                                 f'{self.model}_{self.latest_model_inx}.pth.tar'),
                                    map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])

        if 'aread' in self.model and ('wo' not in self.model):
            for epoch_i in range(self.epoch):
                self.train_aread(model, criterion, optimizer, epoch_i)
                result_dict = self.test(valid_data_loader, model, mode='valid')
                wandb.log(result_dict)
                print(f'validation: auc: {result_dict["total_auc"]:.4f}, loss: {result_dict["total_loss"]:.4f}')
                if result_dict.get("mean_auc") is not None:
                    print(f'validation: mean_auc: {result_dict["mean_auc"]:.4f}, '
                          f'mean_loss: {result_dict["mean_loss"]:.4f}')
                if not self.is_continuable(model, result_dict, epoch_i, optimizer):
                    break
            for d in range(self.n_domain):
                print(f'hierarchical cluster for domain {d}:')
                try:
                    model.print_domain_mask(model.domain_mask[d])
                except:
                    print('mask print error for domain', d)
            wandb.log({'domain_mask': model.domain_mask})

            # mini_batch = 20
            # final_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.final_lr,
            #                                    betas=(0.9, 0.99), eps=1e-8, weight_decay=self.config.wd)
            # for epoch_i in range(self.epoch):
            #     # train final gate
            #     train_domain_batch_seq_len = len(self.train_domain_batch_seq)//mini_batch  # 减少每次循环的样本数
            #     if (epoch_i+1) % mini_batch == 0:
            #         train_domain_batch_seq = self.train_domain_batch_seq[train_domain_batch_seq_len * (epoch_i % mini_batch):]
            #     else:
            #         train_domain_batch_seq = self.train_domain_batch_seq[
            #                                    train_domain_batch_seq_len * (epoch_i % mini_batch):
            #                                    train_domain_batch_seq_len * ((epoch_i + 1) % mini_batch)]
            #     self.train_aread_final(model, criterion, optimizer, train_domain_batch_seq, epoch_i)
            #     result_dict = self.test(valid_data_loader, model, mode='valid')
            #     wandb.log(result_dict)
            #     print(f'validation: auc: {result_dict["total_auc"]:.4f}, loss: {result_dict["total_loss"]:.4f}')
            #     if not self.is_continuable(model, result_dict, epoch_i, final_optimizer):
            #         break
            #
            # print('loading best model...')
            # checkpoint = torch.load(self.save_model_path, map_location=self.device)
            # model.load_state_dict(checkpoint['state_dict'])
            # result_dict = self.test(test_data_loader, model, mode='test', aread_final=True)
            # wandb.log(result_dict)
            # wandb.log({'epoch_i': epoch_i})
            # print('test: ', list(result_dict.items()))
        else:
            for epoch_i in range(self.epoch):
                self.train(train_data_loader, model, criterion, optimizer, epoch_i)
                result_dict = self.test(valid_data_loader, model, mode='valid')
                wandb.log(result_dict)
                print(f'validation: auc: {result_dict["total_auc"]:.4f}, loss: {result_dict["total_loss"]:.4f}')
                if result_dict.get("mean_auc") is not None:
                    print(f'validation: mean_auc: {result_dict["mean_auc"]:.4f}, '
                          f'mean_loss: {result_dict["mean_loss"]:.4f}')
                if not self.is_continuable(model, result_dict, epoch_i, optimizer):
                    break

        print(f'loading best model {self.save_model_path}...')
        checkpoint = torch.load(self.save_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        result_dict = self.test(test_data_loader, model, mode='test')
        wandb.log(result_dict)
        wandb.log({'epoch_i': epoch_i})
        print('test: ', list(result_dict.items()))
        if result_dict.get("mean_auc") is not None:
            print(f'test: mean_auc: {result_dict["mean_auc"]:.4f}, '
                  f'mean_loss: {result_dict["mean_loss"]:.4f}')


class MamdrRun(Run):
    def __init__(self, config):
        super(MamdrRun, self).__init__(config)
        self.meta_weights, self.domain_weights = None, None

    @staticmethod
    def merge_weights(shared_weights, specific_weights):
        merged_weights = {}
        for name in shared_weights:
            # 直接使用张量加法合并对应的权重
            merged_weights[name] = shared_weights[name] + specific_weights[name]
        return merged_weights

    @staticmethod
    def copy_model_weights(model_weights):
        """clone the weights of the model"""
        tmp_key = list(model_weights.keys())[0]
        if isinstance(model_weights[tmp_key], torch.Tensor):
            original_weights = {name: model_weights[name].clone() for name in model_weights}
        else:
            raise ValueError('Model weights should be a dict of tensors')
        return original_weights

    def get_data(self):
        print('========Reading data========')
        cols, data = self.read_split_data(self.preprocess_path, None)
        print('after multi-hot features flatten')
        self.train_data_loader = self.convert2domain_data_loader(data[0], cols, mode='train')
        self.valid_data_loader = self.convert2domain_data_loader(data[1], cols, mode='valid')
        self.test_data_loader = self.convert2domain_data_loader(data[2], cols, mode='test')
        print('\n========Finish reading data========')
        return self.train_data_loader, self.valid_data_loader, self.test_data_loader

    def train_on_domain_sequence(self, model, criterion, optimizer, train_domain_batch_seq):
        model.train()
        loss_sum = 0
        log_interval = 204800//self.config.bs
        tk0 = tqdm.tqdm(train_domain_batch_seq, smoothing=0, mininterval=30)
        for i, d in enumerate(tk0):
            X, y = self.get_domain_data(d)
            pred = model(X)
            loss = criterion(pred.squeeze(), y.squeeze().float())
            loss = loss + model.get_regularization_loss(device=self.device)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=format(loss_sum / log_interval, '.4f'))
                wandb.log({'train_loss': (loss_sum / log_interval)})
                loss_sum = 0

    def train(self, model, criterion, optimizer, epoch_i):
        def shuffle_random_domain(domain_batch_seq):
            unique, counts = np.unique(np.array(domain_batch_seq), return_counts=True)
            sorted_indices = np.argsort(unique)
            shuffled_indices = np.random.permutation(sorted_indices)
            shuffled_seq = np.concatenate([np.repeat(unique[i], counts[i]) for i in shuffled_indices])
            return shuffled_seq, unique[sorted_indices], counts[sorted_indices]

        print('Training Epoch {}:'.format(epoch_i + 1))
        start_time = time.time()

        # Shuffle meta sequence, get all domains and their counts
        shuffle_train_domain_sequence, domain_list, domain_batch_cnt = shuffle_random_domain(self.train_domain_batch_seq)
        domain_batch_cnt_dict = dict(zip(domain_list, domain_batch_cnt))

        # Update Shared
        print('Update Shared...')
        model.set_model_meta_parms(self.meta_weights)
        self.train_on_domain_sequence(model, criterion, optimizer, shuffle_train_domain_sequence)

        # Apply grads
        self.meta_weights = model.update_meta_weight(self.meta_weights, meta_lr=self.config.mamdr_meta_lr)

        # Update specific
        print('\nUpdate Specific...')
        for d in domain_list:
            # sample support domains
            candidate_domains = domain_list[domain_list != d]
            aux_idxs = np.random.choice(candidate_domains, size=self.config.mamdr_aux_sample_num, replace=False)
            aux_idxs = np.append(aux_idxs, d)

            merged_weights = self.merge_weights(self.meta_weights, self.domain_weights[d])

            for aux_idx in aux_idxs:
                print(f"Support Domain: {aux_idx}, Query Domain: {d}")
                # Set the Meta Weights
                model.set_model_meta_parms(merged_weights)

                self.train_on_domain_sequence(model, criterion, optimizer,
                                              train_domain_batch_seq=np.repeat(aux_idx, domain_batch_cnt_dict[aux_idx]))

                # Regularize on target domain
                self.train_on_domain_sequence(model, criterion, optimizer,
                                              train_domain_batch_seq=np.repeat(d,
                                                                               domain_batch_cnt_dict[d]))

                # Apply grads
                model.update_meta_weight(self.domain_weights[d], merged_weights,
                                         meta_lr=self.config.mamdr_meta_lr)
                merged_weights = self.merge_weights(self.meta_weights, self.domain_weights[d])
        end_time = time.time()
        print(f"\nEpoch {epoch_i} Time: {end_time - start_time:.2f}s\n"+'-'*50)

    def test(self, val_test_domain_batch_seq, model, mode='valid'):
        print('Evaluating:')
        model.eval()
        targets, predicts, domains = [], [], []
        prev_domain_idx = -1

        val_test_shared_weights = self.copy_model_weights(self.meta_weights)

        unique, counts = np.unique(np.array(val_test_domain_batch_seq), return_counts=True)
        sorted_indices = np.argsort(unique)
        # sorted_unique, sorted_counts = unique[sorted_indices], counts[sorted_indices]
        sorted_seq = np.concatenate([np.repeat(unique[i], counts[i]) for i in sorted_indices])
        # val_test_domain_batch_cnt_dict = dict(zip(sorted_unique, sorted_counts))

        with torch.no_grad():
            for d in tqdm.tqdm(sorted_seq, smoothing=0, mininterval=60):
                if prev_domain_idx != d:
                    model.set_model_meta_parms(self.merge_weights(val_test_shared_weights,
                                                                  self.copy_model_weights(self.domain_weights[d])))
                    prev_domain_idx = d

                X, y = self.get_domain_data(d, mode=mode)
                pred = model(X)

                targets.append(y.squeeze().cpu().numpy())
                predicts.append(pred.squeeze().cpu().numpy())
                domains.append(X[:, self.domain_idx].cpu().numpy())

        targets = np.concatenate(targets)
        predicts = np.concatenate(predicts)
        domains = np.concatenate(domains)

        result_dict = dict()
        result_dict['total_auc'] = roc_auc_score(targets, predicts)
        result_dict['total_loss'] = log_loss(targets, predicts)
        if self.config.is_evaluate_multi_domain:
            result_dict.update(self.evaluate_multi_domain(targets, predicts, domains))

        return result_dict

    def main(self):
        train_data_loader, valid_data_loader, test_data_loader = self.get_data()
        self.train_data_generator = [iter(self.train_data_loader[d]) for d in range(self.n_domain)]
        self.valid_data_generator = [iter(self.valid_data_loader[d]) for d in range(self.n_domain)]
        self.test_data_generator = [iter(self.test_data_loader[d]) for d in range(self.n_domain)]

        model = self.get_model()
        # Save Meta Weight
        self.meta_weights = model.get_meta_weights()
        self.domain_weights = {d: model.get_meta_weights() for d in range(self.n_domain)}

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     betas=(0.9, 0.99), eps=1e-8, weight_decay=self.config.wd)
        criterion = torch.nn.BCELoss(reduction='mean')
        for epoch_i in range(self.epoch):
            self.train(model, criterion, optimizer, epoch_i)
            result_dict = self.test(self.valid_domain_batch_seq, model, mode='valid')
            wandb.log(result_dict)
            print(f'validation: auc: {result_dict["total_auc"]:.4f}, loss: {result_dict["total_loss"]:.4f}')
            if result_dict.get("mean_auc") is not None:
                print(f'validation: mean_auc: {result_dict["mean_auc"]:.4f}, '
                      f'mean_loss: {result_dict["mean_loss"]:.4f}')
            if not self.is_continuable(model, result_dict, epoch_i, optimizer):
                break

        print('loading best model...')
        checkpoint = torch.load(self.save_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        result_dict = self.test(self.test_domain_batch_seq, model, mode='test')
        wandb.log(result_dict)
        wandb.log({'epoch_i': epoch_i})
        print('test: ', list(result_dict.items()))

