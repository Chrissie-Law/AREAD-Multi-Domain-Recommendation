#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import random
import argparse
import config
from preprocess import DataPreprocessing
from run import Run


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='aread')
    parser.add_argument('--dataset_name', default='aliccp')
    parser.add_argument('--base_model', default='mmoe')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--is_set_seed', type=int, default=1)  # If set to 0, the value of args.seed will be ignored
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--prepare2train_month', type=int, default=12)  # Period of the sample from the Amazon dataset
    parser.add_argument('--domain_filter', default=None)
    parser.add_argument("--group_strategy", default='dcn_3groups_kl')  # Will be ignored if the model is AREAD or ADL
    # for AREAD
    parser.add_argument('--update_lr', type=float, default=1e-2)
    parser.add_argument('--aug_ratio', type=float, default=0.1)
    parser.add_argument('--warm_up_interval', type=int, default=100)
    parser.add_argument('--regroup_interval', type=int, default=2000)
    parser.add_argument('--regroup_update_step', type=int, default=5)
    parser.add_argument('--regroup_eval_step', type=int, default=5)
    parser.add_argument('--candidate_mask_num', type=int, default=10)
    parser.add_argument('--random_modify_sigma', type=float, default=0.2)  # Used to gradually reduce the randomness of the initial candidate mask as training progresses
    parser.add_argument('--init_active_percent', type=float, default=0.7)
    args = parser.parse_args()

    if args.is_set_seed == 0:
        # Generate a unique seed based on all args parameters
        args.seed = hash(frozenset(vars(args).items())) % 10000
        args.is_set_seed = 1
        print('set args.seed:', args.seed)

    for key, value in vars(config).items():
        if key not in vars(args) and not key.startswith('__'):
            setattr(args, key, value)
    print('args:', type(args), args.__dict__)

    if args.is_set_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.data_path = os.path.join(args.data_path, args.dataset_name)
    args.save_path = os.path.join(args.save_path, args.dataset_name)
    return args


if __name__ == '__main__':
    config = load_config()
    datapre = DataPreprocessing(config.data_path, dataset_name=config.dataset_name, domains=[],
                                prepare2train_month=config.prepare2train_month,
                                is_aug=('aread' in config.model), aug_ratio=config.aug_ratio)
    datapre.main()
    datapre.update_config(config)

    print('============Model Training============')
    print(f'model:{config.model}, lr:{config.lr}, bs:{config.bs}, ',
          f'embed_dim:{config.embed_dim}, gpu:{config.gpu}, epoch:{config.epoch}, seed:{config.seed}, '
          f'dataset_name:{config.dataset_name}, strategy:{config.group_strategy}')
    Run(config).main()
