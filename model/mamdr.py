#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Paper: Luo, Linhao, et al. "MAMDR: A model agnostic learning framework for multi-domain recommendation."
          IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 2023.
# @Code : https://github.com/RManLuo/MAMDR
"""

import torch
from torch import nn, flatten
from model.layer import BaseModel, FeaturesLinear, MultiLayerPerceptron
from model.layer import FactorizationMachine
import copy


class MAMDR(BaseModel):
    """
    A pytorch implementation of MAMDR, basemodel is mlp.
    """
    def __init__(self, one_hot_feature_dims, embed_dim, multi_hot_dict, mlp_dims, dropout=0.2,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5):
        super(MAMDR, self).__init__(one_hot_feature_dims, embed_dim, multi_hot_dict,
                                    l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'mamdr'  # version: mlp_meta_mamdr_finetune

        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=True)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp.named_parameters()), l2=l2_reg_dnn)

        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        embed_x = self.embedding(x)
        mlp_input = flatten(embed_x, start_dim=1)

        y = self.output_layer(self.linear(mlp_input) + self.mlp(mlp_input))
        return y.squeeze(1)

    def get_meta_weights(self):
        return {name: param.clone() for name, param in self.named_parameters()}

    def set_model_meta_parms(self, tmp_meta_weights):
        with torch.no_grad():  # Temporarily sets all of the requires_grad flags to false
            for name, param in self.named_parameters():
                if name in tmp_meta_weights:
                    param.copy_(tmp_meta_weights[name])

    def update_meta_weight(self, update_vars, merged_weights=None, meta_lr=0.1):
        new_vars = self.get_meta_weights()
        if merged_weights is not None:
            old_vars = merged_weights
        else:
            old_vars = update_vars
        for name in update_vars.keys():
            update_vars[name] += (new_vars[name] - old_vars[name]) * meta_lr

        return update_vars
