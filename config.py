#!/usr/bin/env python
# -*- coding: utf-8 -*-
use_cuda = 1
gpu = 0
data_path = "dataset"
save_path = "save"
itemid_all = 1368287
seq_maxlen = 5
early_stop = 2
is_increment = 0
is_evaluate_multi_domain = 1
embed_dim = 16
bs = 512
epoch = 10
wd = 0.00000001
is_gauc = 0
domain_positive_label = [35801, 144419, 390519, 1505239, 46542, 205831, 1315090, 964, 701666, 10261, 320500, 914,
                         1232366, 103986, 91991, 3097, 163525, 68118, 309269, 288912, 433660, 554954, 522193,
                         319880, 20878]

# common used mlp_dims for dcn, dcnv2, autoint
mlp_dims = (256, 128, 64)

# common used tower_dims for pepnet, epnet, epnet-single, star, adl
tower_dims = (256, 128, 64, 32)

# autoint
use_atten = True
atten_embed_dim = 64
att_layer_num = 3
att_head_num = 2
att_res = True

# dcn & dcnv2
use_dcn = True
n_cross_layers = 3

# mmoe
mmoe_n_expert = 4
mmoe_expert_dims = (256, 128, 64)
mmoe_tower_dims = (64, 32)

# ple
ple_n_expert_specific = 2
ple_n_expert_shared = 2
ple_expert_dims = ((256, 128), (64,))
ple_tower_dims = (64, 32)


# hinet
sei_dims = [64, 32]

# adl
dlm_iters = 3

# aread
aread_tower_dims = ((64, 32), (32, 16), (16, 8))

domain_size = {
    "amazon": [69360, 282546, 776105, 3001846, 88496, 449031, 2859592, 1893, 1437340, 16454, 601698, 1802, 2416380,
               197170, 202176, 6931, 317131, 132650, 602500, 585227, 845268, 1107407, 997451, 623565, 44843],
    "aliccp": [2695782, 1433175, 925817, 584726, 461755, 358265, 166869, 113621, 78692, 65313, 54483, 45808,
               40975, 37939, 34079, 31703, 29551, 27084, 25027, 23464, 21764, 19857, 18390, 16712, 15852, 14914,
               13653, 12265, 11179, 9760]
}

domain2group_org_dict = {
    'amazon': {
        "dcn_3groups_kl": [0, 1, 0, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    },
    'aliccp': {
        "dcn_3groups_kl": [1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 2, 2, 2, 1, 1, 1, 1],
    }
}

