#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import re
from model.layer import BaseModel, MultiLayerPerceptron, CrossNetwork
from model.ple import PLE, CGC
from model.mmoe import MMoE


class AREAD(BaseModel):
    """
    Adaptive REcommendation for All Domains (AREAD)
    """
    def __init__(self, one_hot_feature_dims, embed_dim, multi_hot_dict, n_tower, n_domain, base_model,
                 expert_dims, tower_dims, domain_idx,
                 domain2group=None, n_cross_layers=3, dropout=0.2, device=None,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=1e-5, l2_reg_cross=1e-5, config=None):
        """
        Initialize the AREAD model.

        :param one_hot_feature_dims: Dimensions for one-hot encoded features.
        :param embed_dim: Embedding dimension for features.
        :param multi_hot_dict: Dictionary for handling multi-hot encoded features.
        :param n_tower: Number of experts in each level of HEI, e.g., [3, 6, 9].
        :param n_domain: Number of distinct domains in the dataset.
        :param base_model: Base model to integrate within the architecture, choosing from MMoE or PLE.
        :param expert_dims: Expert layers in HEI.
        :param tower_dims: Output tower network layers, e.g., [[64, 32], [32, 16], [16, 8]].
        :param domain_idx: Indexes of domain in input features.
        :param domain2group: Optional, used for domain mask initialization.
        :param n_cross_layers: Number of layers in the cross-network part.
        :param dropout: Dropout rate.
        :param device: Device on which to run the model (e.g., 'cuda', 'cpu').
        :param l2_reg_embedding: L2 regularization strength for embeddings.
        :param l2_reg_linear: L2 regularization strength for linear model components.
        :param l2_reg_dnn: L2 regularization strength for deep neural network components.
        :param l2_reg_cross: L2 regularization strength for cross network components.
        :param config: Configuration object containing additional model settings and hyperparameters.
        """

        super(AREAD, self).__init__(one_hot_feature_dims, embed_dim, multi_hot_dict,
                                    l2_reg_embedding=l2_reg_embedding, l2_reg_linear=l2_reg_linear)
        self.model_name = 'aread'
        self.base_model = base_model

        self.domain_idx = domain_idx
        self.n_tower = n_tower  # [tower num for each level]
        self.n_level = len(n_tower)
        self.edge_num = n_tower[0] + np.sum([n_tower[l-1]*n_tower[l] for l in range(1, self.n_level)]) + n_tower[-1]
        self.n_domain = n_domain
        self.tower_dims = tower_dims
        self.bottom_level = len(expert_dims)
        self.device = device
        self.domain2group = np.array([domain2group[d] for d in range(n_domain)]
                                     ) if domain2group is not None else None  # array of domain2group
        self.domain_mask = [None for _ in range(self.n_domain)]  # hierarchical tower mask for each domain
        self.candidate_domain_mask = None  # Used to store candidate domain masks during the clustering update process, with each domain having Z temporary masks generated as candidates
        self.tower2cluster = [[None for _ in range(self.n_tower[l])] for l in range(self.n_level)]
        self.model_state = None  # Used to store the model state
        self.domain_tower_gate_values = None  # Used to store a list of gate values for each tower, for subsequent mask generation
        self.tmp_tower_gate_values = [[None for _ in range(self.n_tower[l])] for l in range(self.n_level)]  # Used to record the gate values of each tower in a single forward step
        self.gate_value_threshold = None  # Used to store the threshold for gate values, for subsequent pruning
        self.eval_loss = None  # Store the evaluation loss for each domain, facilitating subsequent mask selection
        self.group_embedding = torch.nn.Embedding(n_tower[0], embed_dim)
        self.final_gate = nn.Sequential(nn.Linear(2*embed_dim, n_tower[-1], bias=False), nn.Softmax(dim=1))
        self.domain_size = np.array(config.domain_size[config.dataset_name])
        self.use_dcn = getattr(config, 'use_dcn', False)
        self.use_atten = getattr(config, 'use_atten', False)

        if self.use_dcn:
            self.cn = CrossNetwork(self.embed_output_dim, config.n_cross_layers)
        if self.use_atten:
            self.build_atten(config, dropout)

        if self.base_model == 'ple':
            self.cgc_layers = nn.ModuleList(
                CGC(i + 1, self.n_level, n_tower[0],
                    config.ple_n_expert_specific,
                    config.ple_n_expert_shared,
                    self.embed_output_dim if i == 0 else expert_dims[i - 1][-1], expert_dims[i], dropout)
                for i in range(self.bottom_level))
            tower_input_dim = expert_dims[-1][-1]

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cgc_layers.named_parameters()),
                l2=l2_reg_dnn)
        elif self.base_model == 'mmoe':
            self.mmoe_experts = nn.ModuleList(MultiLayerPerceptron(self.embed_output_dim, expert_dims, dropout,
                                                                   output_layer=False) for _ in
                                              range(config.mmoe_n_expert))
            self.mmoe_gates = nn.ModuleList([
                nn.Sequential(nn.Linear(self.embed_output_dim, config.mmoe_n_expert),
                              nn.Softmax(dim=1))
                for _ in range(self.n_tower[0])])
            tower_input_dim = expert_dims[-1]

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0],
                       self.mmoe_experts.named_parameters()), l2=l2_reg_dnn)

        towers_list, tower_gates_list = [], []
        for l in range(self.n_level):
            towers_list.append(nn.ModuleList(
                MultiLayerPerceptron(tower_input_dim, tower_dims[l], dropout, output_layer=False)
                for _ in range(self.n_tower[l])))
            if l != 0:
                tower_gates_list.append(nn.ModuleList([
                    nn.Sequential(nn.Linear(2*embed_dim, self.n_tower[l-1]))
                    for _ in range(self.n_tower[l])]))
            tower_input_dim = tower_dims[l][-1]
        self.towers = nn.ModuleList(towers_list)
        self.tower_gates = nn.ModuleList(tower_gates_list)

        self.towers_linear = nn.ModuleList(nn.Linear(self.embed_output_dim + tower_dims[-1][-1], 1, bias=False)
                                           for _ in range(self.n_tower[-1]))
        self.output_layers = nn.ModuleList([nn.Sigmoid() for _ in range(n_tower[-1])])

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.towers.named_parameters()), l2=l2_reg_dnn)
        if self.use_dcn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cn.named_parameters()), l2=l2_reg_cross)

    def forward(self, x, mode='wo_mask', targets=None, memory_gate_value=False,
                domain_i=None, current_mask=None, tmp_memory_gate_value=False):
        embed_x = self.embedding(x, squeeze_dim=False)
        domain_embed = embed_x[:, self.domain_idx, :].squeeze(1)
        embed_x = embed_x.flatten(start_dim=1)

        # other predict
        linear_out = self.linear(embed_x)
        if self.use_dcn:
            cn_out = self.cn(embed_x)
        if self.use_atten:
            atten_out = self.atten_forward(embed_x)

        if self.base_model == 'ple':
            ple_inputs = [embed_x] * (self.n_tower[0] + 1)
            ple_outs = []
            for l in range(self.bottom_level):
                ple_outs = self.cgc_layers[l](ple_inputs)  # ple_outs[i]: [batch_size, expert_dims[-1]]
                ple_inputs = ple_outs
            tower_inputs = [ple_outs[t] for t in range(self.n_tower[0])]
        elif self.base_model == 'mmoe':
            expert_outs = [expert(embed_x).unsqueeze(1) for expert in self.mmoe_experts]
            expert_outs = torch.cat(expert_outs, dim=1)  # [batch_size, n_expert, expert_dims[-1]]
            gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.mmoe_gates]  # gate_out[i]: [batch_size, n_expert, 1]
            tower_inputs = [torch.sum(torch.mul(gate_out, expert_outs), dim=1) for gate_out in gate_outs]

        # Prediction
        if mode == 'wo_mask':
            group_embed = torch.zeros_like(
                domain_embed)  # Since no mask is needed, the group's significance is diminished
            # Pass data from all domains through all networks
            gate_inputs = torch.cat([domain_embed, group_embed], dim=1)
            if memory_gate_value:
                tmp_tower_gate_values = [[None for _ in range(self.n_tower[l])] for l in range(self.n_level)] + [
                    [None for _ in range(self.n_tower[-1])]]
            for l in range(self.n_level):
                if l > 0:
                    # The first layer's tower does not forward through the gate as it has already been integrated in the bottom
                    tower_inputs = []
                    for t in range(self.n_tower[l]):
                        gate_out = F.softmax(self.tower_gates[l - 1][t](gate_inputs), dim=1)  # shape: [batch_size, n_tower[l-1]]
                        tower_outs_weight = torch.mul(gate_out.unsqueeze(-1), tower_outs)
                        tower_inputs.append(torch.sum(tower_outs_weight, dim=1))
                        if memory_gate_value:
                            tmp_tower_gate_values[l][t] = gate_out.detach().clone()
                if l == self.n_level - 1:
                    if self.use_dcn:
                        tower_outs = [torch.cat([cn_out, self.towers[l][t](tower_inputs[t])], dim=1)
                                      for t in range(self.n_tower[l])]
                    else:
                        tower_outs = [self.towers[l][t](tower_inputs[t]) for t in range(self.n_tower[l])]
                    # concat with cn, shape: n_tower[l], batch_size, tower_dims[l][-1]+embed_output_dim
                else:
                    tower_outs = torch.stack([self.towers[l][t](tower_inputs[t]) for t in range(self.n_tower[l])],
                                             dim=1)  # shape: [batch_size, n_tower[l], tower_dims[l][-1]]
            y_stack = torch.stack([self.output_layers[t](self.towers_linear[t](tower_outs[t]) + linear_out)
                                   for t in range(self.n_tower[-1])], dim=0)
            y = torch.mean(y_stack, dim=0)
            if memory_gate_value:
                if domain_i is None:
                    for d in range(self.n_domain):
                        domain_ids = x[:, self.domain_idx]
                        mask = (domain_ids == d).squeeze()
                        for l in range(1, self.n_level):
                            for t in range(self.n_tower[l]):
                                self.domain_tower_gate_values[d][l][t].append(
                                    torch.mean(tmp_tower_gate_values[l][t][mask], dim=0))
                else:
                    for l in range(1, self.n_level):
                        for t in range(self.n_tower[l]):
                            self.domain_tower_gate_values[domain_i][l][t].append(
                                torch.mean(tmp_tower_gate_values[l][t], dim=0))

            return y
        elif mode == 'with_mask':
            # domain data forward based on its mask
            ys = []
            domained_targets = []
            domain_ids = x[:, self.domain_idx]
            total_tower_inputs = tower_inputs
            for d in range(self.n_domain):
                mask = (domain_ids == d).squeeze()
                tower_inputs = [total_tower_inputs[t][mask] for t in range(self.n_tower[0])]
                gate_inputs = torch.cat(tower_inputs, dim=1)

                current_domain_mask = self.domain_mask[d]
                domain_other_outs = [other_out[mask] for other_out in other_outs]

                y_stack = self.hier_tower_mask_forward(d, tower_inputs, gate_inputs, cn_out, linear_out,
                                                       current_domain_mask, memory_gate_value)

                ys.append(torch.mean(y_stack, dim=0))
                domained_targets.append(targets[mask])

            return torch.cat(ys, dim=0), torch.cat(domained_targets, dim=0)
        elif mode == 'domain_with_mask':
            current_mask = self.domain_mask[domain_i] if current_mask is None else current_mask
            group_embed = self.group_embedding(torch.nonzero(current_mask[0])[:, 1])
            if group_embed.shape[0] > 1:
                group_embed = torch.mean(group_embed, dim=0, keepdim=True)
            group_embed = group_embed.expand(x.shape[0], -1)
            gate_inputs = torch.cat([domain_embed, group_embed], dim=1)
            y_stack = self.hier_tower_mask_forward(domain_i, tower_inputs, gate_inputs, cn_out, linear_out,
                                                   current_mask, memory_gate_value, tmp_memory_gate_value)
            y = torch.mean(y_stack, dim=0)
            return y
        elif mode == 'domain_mask_bagging':
            current_mask = self.domain_mask[domain_i] if current_mask is None else current_mask
            group_embed = self.group_embedding(torch.nonzero(current_mask[0])[:, 1])
            if group_embed.shape[0] > 1:
                group_embed = torch.mean(group_embed, dim=0, keepdim=True)
            group_embed = group_embed.expand(x.shape[0], -1)
            gate_inputs = torch.cat([domain_embed, group_embed], dim=1)
            y_stack = self.hier_tower_mask_forward(domain_i, tower_inputs, gate_inputs, cn_out, linear_out,
                                                   current_mask, memory_gate_value, tmp_memory_gate_value)
            return y_stack
        elif mode == 'domain_mask_final':
            with torch.no_grad():
                # Freeze the preceding parameters when training the final gate
                current_mask = self.domain_mask[domain_i] if current_mask is None else current_mask
                group_embed = self.group_embedding(torch.nonzero(current_mask[0])[:, 1])
                if group_embed.shape[0] > 1:
                    group_embed = torch.mean(group_embed, dim=0, keepdim=True)
                group_embed = group_embed.expand(x.shape[0], -1)
                gate_inputs = torch.cat([domain_embed, group_embed], dim=1)
                y_stack = self.hier_tower_mask_forward(domain_i, tower_inputs, gate_inputs,
                                                       cn_out, linear_out, current_mask,
                                                       memory_gate_value)
            final_gate_out = self.final_gate(torch.cat([domain_embed, group_embed], dim=1).detach()
                                             ) * current_mask[-1].squeeze(1)
            final_gate_mask = final_gate_out / (final_gate_out.sum(dim=1, keepdim=True) + 1e-8)
            y = torch.sum(torch.mul(y_stack.transpose(0, 1).squeeze(-1), final_gate_mask), dim=1)
            return y

    def hier_tower_mask_forward(self, d, tower_inputs, gate_inputs, domain_cn_out, domain_linear_out,
                                single_domain_mask, memory_gate_value=False, tmp_memory_gate_value=False):
        batch_size = domain_linear_out.shape[0]
        for l in range(self.n_level):
            last_level_active_tower = this_level_active_tower if l > 0 else None  # len: n_tower[l-1]
            this_level_active_tower = torch.any(single_domain_mask[l], dim=0)  # len: n_tower[l]
            if l > 0:
                tower_inputs = []
                for t in range(self.n_tower[l]):
                    if not this_level_active_tower[t]:
                        tower_inputs.append(torch.zeros(batch_size, self.tower_dims[l][-1],
                                                        dtype=torch.float32, device=self.device))
                        if memory_gate_value:
                            self.domain_tower_gate_values[d][l][t].append(
                                torch.zeros(self.n_tower[l - 1], dtype=torch.float32, device=self.device))
                        if tmp_memory_gate_value:
                            self.tmp_tower_gate_values[l][t] = torch.zeros(self.n_tower[l - 1],
                                                                           dtype=torch.float32, device=self.device)
                        continue
                    gate_out = F.softmax(self.tower_gates[l - 1][t](gate_inputs), dim=1)  # shape: [batch_size, n_tower[l-1]]
                    gate_out_masked = gate_out * single_domain_mask[l][:, t]
                    sums = gate_out_masked.sum(dim=1, keepdim=True) + 1e-8
                    gate_out_rescaled = gate_out_masked / sums
                    # Rescale to ensure the gate sum equals 1
                    tower_outs_weight = torch.mul(gate_out_rescaled.unsqueeze(-1), tower_outs)
                    tower_inputs.append(torch.sum(tower_outs_weight, dim=1))

                    if tmp_memory_gate_value:
                        self.tmp_tower_gate_values[l][t] = (
                            torch.mean(gate_out * single_domain_mask[l][:, t], dim=0)).detach().clone()
                    if memory_gate_value:
                        self.domain_tower_gate_values[d][l][t].append(
                            (torch.mean(gate_out * single_domain_mask[l][:, t], dim=0)).detach().clone())
            else:
                tower_inputs = [tower_inputs[t] if this_level_active_tower[t]
                                else torch.zeros(batch_size, self.tower_dims[l][-1],
                                                 dtype=torch.float32, device=self.device)
                                for t in range(self.n_tower[0])]

            # shape: [batch_domain_size, n_tower[l], tower_dims[l][-1]]
            if l == self.n_level - 1:
                tower_outs = [
                    self.output_layers[i](
                        self.towers_linear[i](
                            torch.cat([domain_cn_out, self.towers[l][i](tower_inputs[i])], dim=1)
                        ) + domain_linear_out).squeeze(-1)
                    for i in range(self.n_tower[l]) if this_level_active_tower[i]
                ]
                # concat with cn, shape: valid n_tower[l], batch_size, tower_dims[l][-1]+embed_output_dim
                try:
                    y_stack = torch.stack(tower_outs, dim=0)  # shape: valid n_tower[-1], batch size, 1
                except RuntimeError:
                    print('RuntimeError')
                    print(f'domain {d} mask: {single_domain_mask}')
                    self.print_domain_mask(single_domain_mask, all_edges=True)
            else:
                tower_outs = torch.stack([self.towers[l][i](tower_inputs[i])
                                          if this_level_active_tower[i] else tower_inputs[i]
                                          for i in range(self.n_tower[l])], dim=1)
        return y_stack  # shape: valid n_tower[-1], batch size

    def add_eval_loss(self, loss_mean, d, mask_z):
        if len(self.eval_loss[d]) <= mask_z:
            self.eval_loss[d].append([loss_mean])
        else:
            self.eval_loss[d][mask_z].append(loss_mean)

    def update_all_mask(self, regroup_times=None, update_mode='best4single_domain'):
        tmp_masks_mean, tmp_masks_std = [], []
        print('\n============Update Mask============')
        if update_mode == 'best4single_domain':
            tmp_mask_num = len(self.candidate_domain_mask[0])
            loss_mean = [[None for _ in range(tmp_mask_num)] for _ in range(self.n_domain)]
            for d in range(self.n_domain):
                for z in range(tmp_mask_num):
                    loss_mean[d][z] = np.mean(self.eval_loss[d][z])
                self.domain_mask[d] = self.candidate_domain_mask[d][np.argmin(loss_mean[d])]
                tmp_masks_mean.append(np.mean(loss_mean[d]))
                tmp_masks_std.append(np.std(loss_mean[d]))
                # print(f'domain {d} mask updated, loss_mean of different masks: {loss_mean[d]}')
            print('regroup_times: ', regroup_times,
                  'current domain mask active ratio: ', self.count_current_active_ratio())
            print(f'loss_mean of different domain masks: {tmp_masks_mean}')
            print(f'loss_std of different domain masks: {tmp_masks_std}')
            tower_active_domain = [[] for _ in range(self.n_tower[1])]
            for d in range(self.n_domain):
                middle_layer_of_d = torch.nonzero(torch.any(self.domain_mask[d][1], axis=0))[:, 0].cpu().numpy()
                for t in middle_layer_of_d:
                    tower_active_domain[t].append(d)
            print(f'active domain num of each tower in the middle layer: {[len(t) for t in tower_active_domain]}')
            print('sample size training each tower in the middle layer: '
                  f'{[sum(self.domain_size[t]) for t in tower_active_domain]}')
            print('============Finish Update Mask============')

    def prun_single_mask(self, d, current_mask, prun_ratio=0.05):
        gate_values = []
        threshold = 1
        for l in range(1, self.n_level):
            gate_values.append(torch.stack(self.tmp_tower_gate_values[l], dim=1))
            if (gate_values[-1] > 1e-8).any():
                # Both the first and last layers are set to zero by default; the threshold is determined by the intermediate layers
                threshold = min(threshold,
                                torch.quantile(gate_values[-1][gate_values[-1] > 1e-8].flatten(), prun_ratio))

        if threshold == 1:
            self.print_domain_mask(current_mask, all_edges=True)
            for i in range(self.n_level):
                print(f'level {i} gate_values: {self.tmp_tower_gate_values[i]}')
            raise ValueError('no valid tmp_tower_gate_values in candidate mask')
        prun_mask = [t >= threshold for t in gate_values]
        before_prun_mask = copy.deepcopy(current_mask)

        for l in range(1, self.n_level):
            current_mask[l] = current_mask[l] & prun_mask[l-1]
        valid_mask = self.validate_mask(current_mask)

        self.tmp_tower_gate_values = [[None for _ in range(self.n_tower[l])] for l in range(self.n_level)]

        return valid_mask if valid_mask[-1].any().item() else before_prun_mask  # 如果裁剪后没有输出，则不更新

    def reset_for_mask_update(self, d=None):
        if d is None:
            self.domain_tower_gate_values = [
                [[[] for _ in range(self.n_tower[l])]
                 for l in range(self.n_level)] + [
                    [[] for _ in range(self.n_tower[-1])]]
                for _ in range(self.n_domain)
            ]
            self.gate_value_threshold = [None for _ in range(self.n_domain)]
            self.candidate_domain_mask = [[] for _ in range(self.n_domain)]
            self.eval_loss = [[] for _ in range(self.n_domain)]
        else:
            # Record the input gate value for each tower using a list, shape: n_level+1, n_tower[l], 0
            self.domain_tower_gate_values[d] = [[[] for _ in range(self.n_tower[l])]
                                                for l in range(self.n_level)] + [
                                                   [[] for _ in range(self.n_tower[-1])]]
            self.gate_value_threshold[d] = None
            self.candidate_domain_mask[d] = []
            self.eval_loss[d] = []

    def mean_domain_tower_gate_values(self, d, get_threshold=None):
        if isinstance(self.domain_tower_gate_values[d][0], list):
            mean_values = []
            for l in range(self.n_level):
                if l == 0:
                    mean_values.append(torch.zeros(1, self.n_tower[0], dtype=torch.float32, device=self.device))
                else:
                    mean_values_layer = []
                    for t in range(self.n_tower[l]):
                        # The shape of tensor unit in the list should be [n_tower[l-1],]
                        if len(self.domain_tower_gate_values[d][l][t]) == 0:
                            mean_values_layer.append(torch.zeros(self.n_tower[l - 1], dtype=torch.float32,
                                                                 device=self.device))
                        else:
                            mean_values_layer.append(torch.mean
                                                     (torch.stack(self.domain_tower_gate_values[d][l][t], dim=0),
                                                      dim=0))
                    mean_values.append(torch.stack(mean_values_layer, dim=1))

            mean_values.append(torch.zeros(self.n_tower[-1], 1, dtype=torch.float32, device=self.device))
            self.domain_tower_gate_values[d] = mean_values

            if get_threshold is not None:
                threshold = 1
                for ts in mean_values[1:-1]:
                    if (ts > 1e-8).any():
                        threshold = min(threshold, torch.quantile(ts[ts > 1e-8].flatten(), 1-get_threshold))
                self.gate_value_threshold[d] = None if threshold == 1 else threshold

    def generate_mask(self, generate_mode='rand', d=None, init_active_percent=0.7, random_modify_sigma=0.2):
        """
        :param generate_mode: 'rand', 'mask_norm_rand', 'max_gate', or 'max_gate_norm_rand'
        :param d: domain id
        :param init_active_percent: initial active/True percentage of units
        :return: a mask for single domain to re-group
        """
        if generate_mode == 'rand':
            is_mask_output = False
            while not is_mask_output:
                mask = self.create_single_full_mask(fill_value=init_active_percent)
                valid_mask = self.validate_mask(mask)
                is_mask_output = valid_mask[-1].any().item()
            tensor_mask = [torch.tensor(valid_mask[l], dtype=torch.bool, device=self.device)
                    for l in range(self.n_level + 1)]
            return tensor_mask
        elif generate_mode == 'mask_norm_rand':
            same_with_origin = True
            original_mask = [self.domain_mask[d][l].cpu().numpy() for l in range(self.n_level + 1)]
            active_edge_num = self.count_active_edge(d_mask=original_mask)
            while same_with_origin:
                mask = []
                rand_percent = min(1, np.abs(np.random.normal(0, random_modify_sigma)))
                if active_edge_num < self.edge_num * rand_percent:  # 有效边较少，需要增加
                    for l in range(self.n_level + 1):
                        # random add the edges in original_mask[l]
                        rand_mask = np.random.rand(*original_mask[l].shape) < rand_percent
                        mask.append(original_mask[l] | rand_mask)
                else:
                    for l in range(self.n_level + 1):
                        # random flip the edges in original_mask[l]
                        rand_mask = np.random.rand(*original_mask[l].shape) < rand_percent
                        mask.append(original_mask[l] ^ rand_mask)

                valid_mask = self.validate_mask(mask)

                same_with_origin = True
                for l in range(self.n_level + 1):
                    if not np.all(valid_mask[l] == original_mask[l]):
                        same_with_origin = False
                        break
                same_with_origin = same_with_origin or (not valid_mask[-1].any().item())  # 最后一层必须有输出

            return [torch.tensor(valid_mask[l], dtype=torch.bool, device=self.device)
                    for l in range(self.n_level + 1)]
        elif generate_mode in ['max_gate', 'max_gate_norm_rand']:
            if not any(self.domain_tower_gate_values):
                raise ValueError('tower_gate_values is None')
            self.mean_domain_tower_gate_values(d, get_threshold=init_active_percent)
            if self.gate_value_threshold[d] is None:
                return self.generate_mask(generate_mode='rand', d=d, init_active_percent=init_active_percent,
                                          random_modify_sigma=random_modify_sigma)
            prun_mask = [t >= self.gate_value_threshold[d] for t in self.domain_tower_gate_values[d]]
            if generate_mode == 'max_gate':
                valid_mask = self.validate_mask(prun_mask)
                if not valid_mask[-1].any().item():
                    raise ValueError(f"mask generated for domain {d} in the 'max_gate' mode has no output")
            else:  # max_gate_norm_rand
                rand_percent = min(1, np.abs(np.random.normal(0, random_modify_sigma)))
                is_mask_output = False
                while not is_mask_output:
                    mask = []
                    for l in range(self.n_level + 1):
                        # 随机翻转一些边
                        rand_tensor = torch.rand(prun_mask[l].shape, device=self.device) < rand_percent
                        mask.append(prun_mask[l] ^ rand_tensor)
                    valid_mask = self.validate_mask(mask)
                    is_mask_output = valid_mask[-1].any().item()
            return valid_mask
        elif generate_mode == 'mask_max_gate':
            if not any(self.domain_tower_gate_values):
                raise ValueError('tower_gate_values is None')
            self.mean_domain_tower_gate_values(d, get_threshold=init_active_percent)
            if self.gate_value_threshold[d] is None:
                prun_mask = self.generate_mask(generate_mode='rand', d=d, init_active_percent=init_active_percent,
                                               random_modify_sigma=random_modify_sigma)
            else:
                prun_mask = [t >= self.gate_value_threshold[d] for t in self.domain_tower_gate_values[d]]

            rand_percent = min(1, np.abs(np.random.normal(0, random_modify_sigma)))
            origin_mask = self.domain_mask[d] if self.domain_mask[d] is not None else prun_mask
            is_nor = ((self.count_active_edge(d_mask=origin_mask)*1. / self.edge_num) > init_active_percent)
            same_with_origin = True
            while same_with_origin:
                mask = []
                for l in range(self.n_level + 1):
                    # Randomly invert the masking state of some gates
                    rand_tensor = torch.rand(prun_mask[l].shape, device=self.device) < rand_percent
                    if is_nor:
                        mask.append((origin_mask[l] | prun_mask[l]) ^ rand_tensor)  # 原来的mask合并最大边之后，随机修改
                    else:
                        mask.append((origin_mask[l] | prun_mask[l]) | rand_tensor)  # mask比较小，随机增加
                valid_mask = self.validate_mask(mask)

                same_with_origin = True
                for l in range(self.n_level + 1):
                    if not torch.all(valid_mask[l] == origin_mask[l]):
                        same_with_origin = False
                        break
                same_with_origin = same_with_origin or (not valid_mask[-1].any().item())
            return valid_mask

    def save_model_state(self):
        params_to_save = ['cn', 'cgc_layers', 'towers', 'tower_gates', 'towers_linear', 'output_layers',
                          'embedding', 'linear', 'reg_loss', 'regularization_weight']

        regex_pattern = '^(' + '|'.join(params_to_save) + ')'
        pattern = re.compile(regex_pattern)

        full_state_dict = self.state_dict()
        selected_state_dict = {k: v for k, v in full_state_dict.items() if pattern.match(k)}
        self.model_state = copy.deepcopy(selected_state_dict)

    def load_model_state(self):
        self.load_state_dict(self.model_state, strict=False)

    def create_single_full_mask(self, fill_value=0):
        """
        :param fill_value: 0 or 1, fill_value in (0, 1) means {fill_value} percentage of units are True
        :return: a list of mask for single domain, filled with fill_value, mask[l] is the input edge mask for level l
        """
        if fill_value == 0:
            full_mask = [np.zeros((1, self.n_tower[0]), dtype=bool),] + [
                np.zeros((self.n_tower[l - 1], self.n_tower[l]), dtype=bool)
                for l in range(1, self.n_level)] + [np.zeros((self.n_tower[-1], 1), dtype=bool),]
        elif fill_value == 1:
            full_mask = [np.ones((1, self.n_tower[0]), dtype=bool),] + [
                np.ones((self.n_tower[l - 1], self.n_tower[l]), dtype=bool)
                for l in range(1, self.n_level)] + [np.ones((self.n_tower[-1], 1), dtype=bool),]
        elif 0 < fill_value < 1:
            full_mask = [np.random.choice([True, False], (1, self.n_tower[0]), p=[fill_value, 1 - fill_value]),] + [
                np.random.choice([True, False], (self.n_tower[l - 1], self.n_tower[l]), p=[fill_value, 1 - fill_value])
                for l in range(1, self.n_level)] + [
                np.random.choice([True, False], (self.n_tower[-1], 1), p=[fill_value, 1 - fill_value]),]
        else:
            raise ValueError('fill_value in mask must be 0 or 1 or (0, 1)')
        return full_mask

    def validate_mask(self, mask, add_input=True, add_output=True, remove_hidden=True):
        """
        mask is a list of mask (array or tensor) for single domain
        :return: a list of mask for single domain, with input and output edges added, and hidden invalid edges removed
        """
        if add_input:
            # If the lowest layer's experts has outputs, it requires input edges.
            for t in range(self.n_tower[0]):
                if mask[1][t, :].any():
                    mask[0][:, t] = True
        if add_output:
            # If the highest layer's experts has inputs, it requires output edges.
            for t in range(self.n_tower[-1]):
                if mask[-2][:, t].any():
                    mask[-1][t, :] = True
        if remove_hidden:
            tower_to_check = [(l, t) for l in range(1, self.n_level) for t in range(self.n_tower[l])]
            while len(tower_to_check) > 0:
                l, t = tower_to_check.pop(0)
                # If the current experts has no inputs from the previous layer, sever its output edges.
                if not mask[l][:, t].any():
                    mask[l + 1][t, :] = False
                # If the current experts has no outputs to the next layer, sever its input edges.
                if not mask[l + 1][t, :].any():
                    last_level_input = []
                    if l > 1:
                        if isinstance(mask[l], torch.Tensor):
                            last_level_input = mask[l][:, t].nonzero()[:, 0].tolist()
                        elif isinstance(mask[l], np.ndarray):
                            last_level_input = np.nonzero(mask[l][:, t])[0].tolist()
                        if len(last_level_input):
                            for t in last_level_input:
                                if (l - 1, t) not in tower_to_check:
                                    tower_to_check.append((l - 1, t))
                    mask[l][:, t] = False
        return mask

    def create_domain_mask(self, cluster_z):
        """
        Initialize the domain_mask based on the results of hierarchical clustering.
        :param cluster_z: Linkage matrix ndarray encoding the hierarchical clustering, shape: (n_domain - n_tower[0], 4)
        """
        self.domain_mask = [self.create_single_full_mask() for _ in range(self.n_domain)]
        clusters = [[i] for i in range(self.n_domain)]
        cluster_exist = [i for i in range(self.n_domain)]
        tower2cluster = [None for _ in range(self.n_level)]
        for i in range(self.n_domain - self.n_tower[0]):
            line = cluster_z[i]
            clusters.append(clusters[int(line[0])] + clusters[int(line[1])])

            cluster_exist.append(i + self.n_domain)
            cluster_exist.remove(int(line[0]))
            cluster_exist.remove(int(line[1]))

            if len(cluster_exist) in self.n_tower:  # When the number of remaining clusters matches a tower layer, assign them to that layer.
                tower2cluster[self.n_tower.index(len(cluster_exist))] = copy.deepcopy(cluster_exist)

        for l in range(self.n_level):
            for t in range(self.n_tower[l]):
                domain_cluster = clusters[tower2cluster[l][t]]  # Extract the domain cluster managed by this tower
                self.tower2cluster[l][t] = domain_cluster
                for d in domain_cluster:
                    # For domain d, set all outgoing edges from tower at layer l to True, and trim invalid connections later.
                    self.domain_mask[d][l + 1][t, :] = True

        valid_mask = [self.validate_mask(mask) for mask in self.domain_mask]

        self.domain_mask = [[torch.tensor(valid_mask[d][l], dtype=torch.bool, device=self.device)
                             for l in range(self.n_level + 1)] for d in range(self.n_domain)]

    def print_domain_mask(self, d_mask=None, d=None, all_edges=False):
        """
        illustrate domain mask
        """
        mask = d_mask if d_mask is not None else self.domain_mask[d]
        if isinstance(mask[0], torch.Tensor):
            mask = [m.cpu().numpy() for m in mask]
        print('level 0 towers:', np.nonzero(mask[0])[1])
        if all_edges:
            for l in range(1, self.n_level):
                print(f'========= level {l} =========')
                ll_input = mask[l]
                for t in range(self.n_tower[l]):
                    if ll_input[:, t].any() > 0:
                        print(f'last level input towers of tower {t}:', np.nonzero(ll_input[:, t])[0])
                    else:
                        print(f'last level input towers of tower {t}: None')
            print('========= level finish =========')
            print('the last level output towers:', np.nonzero(mask[-1])[0])
        else:
            for l in range(1, self.n_level):
                print(f'level {l} used last level towers:', np.nonzero(np.any(mask[l], axis=1))[0])
            print('the last level output towers:', np.nonzero(mask[-1])[0])

    def count_current_active_ratio(self):
        ratio_sum = 0
        for d in range(self.n_domain):
            active_edge_num = self.count_active_edge(d=d)
            ratio_sum += active_edge_num*1./ self.edge_num
        return ratio_sum / self.n_domain

    def count_active_edge(self, d=None, d_mask=None):
        mask = d_mask if d_mask is not None else self.domain_mask[d]
        cnt = 0
        if isinstance(mask[0], torch.Tensor):
            for m in mask:
                cnt += torch.sum(m).cpu().item()
        else:
            for m in mask:
                cnt += np.sum(m)
        return cnt
