#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Paper: Li, Jinyun, et al. "ADL: Adaptive Distribution Learning Framework for Multi-Scenario CTR Prediction."
          Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2023.
"""

import torch
import torch.nn as nn
from torch import Tensor
from model.layer import BaseModel, CrossNetwork, DNN, MultiLayerPerceptron
from torch.nn.modules.batchnorm import _NormBase
import torch.nn.functional as F


class ADL(BaseModel):
    def __init__(self, one_hot_feature_dims, embed_dim,
                 multi_hot_dict, n_tower, tower_dims, domain_idx=None, dropout=0.2,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5,
                 dlm_iters=3, dlm_update_rate=0.9, device=None, config=None):
        super(ADL, self).__init__(one_hot_feature_dims, embed_dim, multi_hot_dict,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'adl'
        self.n_tower = n_tower
        self.domain_idx = domain_idx
        self.device = device
        self.cluster_num = n_tower
        self.dlm_iters = dlm_iters
        self.dlm_update_rate = dlm_update_rate
        self.cluster_centers = torch.randn((self.cluster_num, self.embed_output_dim)).to(self.device)
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)
        print("ADL cluster_num:", self.cluster_num, "dlm_iters:", self.dlm_iters,
              "dlm_update_rate:", self.dlm_update_rate)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        self.domain_mlps = nn.ModuleList([
            MultiLayerPerceptron(self.embed_output_dim, tower_dims, dropout, output_layer=False)
            for _ in range(n_tower)])

        self.domain_mlps_linears = nn.ModuleList([nn.Linear(tower_dims[-1], 1)
                                                  for _ in range(n_tower)])

        self.shared_mlps = MultiLayerPerceptron(self.embed_output_dim, tower_dims, dropout, output_layer=False)
        self.shared_mlps_linear = nn.Linear(tower_dims[-1], 1)

        self.output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower)])

        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.domain_mlps.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_mlps.named_parameters()), l2=l2_reg_dnn)

    def DLM_routing(self, embed_x):
        """
        implement DLM routing by updating cluster centers
        @param embed_x: [batch_size, embed_dim]
        @return: distribution_coefficients [batch_size, cluster_num]
        """
        with torch.no_grad():
            for t in range(self.dlm_iters):
                similarity_scores = torch.matmul(embed_x, self.cluster_centers.t())
                distribution_coefficients = F.softmax(similarity_scores, dim=1)  # [batch_size, cluster_num]

                weighted_sum = torch.matmul(distribution_coefficients.t(), embed_x)  # [cluster_num, embed_dim]
                tmp_cluster_centers = F.normalize(weighted_sum, p=2, dim=1)

            self.cluster_centers = F.normalize(
                self.dlm_update_rate * self.cluster_centers + (1 - self.dlm_update_rate) * tmp_cluster_centers, p=2, dim=1)
        return distribution_coefficients

    def forward(self, x):
        embed_x = self.embedding(x, squeeze_dim=True)

        distribution_coefficients = self.DLM_routing(embed_x)

        x_to_tower = torch.argmax(distribution_coefficients, dim=1)

        ys_tensor = torch.zeros((embed_x.size(0), 1), dtype=torch.float, device=self.device)

        shared_mlp_input = embed_x
        shared_mlp_out = self.shared_mlps(shared_mlp_input)

        other_outs = [self.linear(embed_x)]
        if self.use_dcn:
            cn_out = self.cn(embed_x)
            other_outs.append(cn_out)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)
            other_outs.append(atten_out)

        for tower_id in range(self.n_tower):
            domain_mlp = self.domain_mlps[tower_id]
            domain_mlp_linears = self.domain_mlps_linears[tower_id]

            mask = (x_to_tower == tower_id)
            domain_mlp_input = embed_x[mask]

            domain_dnn_out = domain_mlp(domain_mlp_input)
            weight_linear = domain_mlp_linears.weight * self.shared_mlps_linear.weight
            bias_linear = domain_mlp_linears.bias + self.shared_mlps_linear.bias

            domain_dnn_logit = F.linear(domain_dnn_out, weight_linear, bias_linear)
            for other in other_outs:
                domain_dnn_logit += other[mask]

            ys_tensor[mask] = self.output_layers[tower_id](domain_dnn_logit)

        return ys_tensor

