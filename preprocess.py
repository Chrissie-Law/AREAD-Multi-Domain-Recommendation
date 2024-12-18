#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import random
import time
from datetime import timedelta, datetime
import re
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import argparse
import pickle
import ast
import gc
from dataset.aliccp.preprocess_ali_ccp import reduce_mem


class DataPreprocessing(object):
    def __init__(self, data_path, dataset_name, domains, k_cores=3, prepare2train_month=6, is_aug=False, aug_ratio=0,
                 downsample_freq_thresh=15, sample_n_domain=30, sample_mode="interval_random",
                 discrete_method="uniform", negative_sampling_ratio=4):
        """
        :param domains: selected domains, only used for amazon dataset
        :param k_cores: interaction frequency threshold
        :param prepare2train_month: time span of the dataset
        :param is_aug: whether to augment the data
        :param aug_ratio: augmentation ratio
        :param downsample_freq_thresh: downsample frequency threshold, only used for aliccp dataset
        :param sample_n_domain: number of domains after preprocessing
        :param sample_mode: sampling mode of domains
        :param discrete_method: method to discretize continuous features
        :param negative_sampling_ratio: negative sampling ratio, only used for cloudtheme dataset
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.domains = domains
        self.k_cores = k_cores
        self.downsample_freq_thresh = downsample_freq_thresh
        self.sample_n_domain = sample_n_domain
        self.sample_mode = sample_mode
        self.discrete_method = discrete_method
        if dataset_name == 'amazon':
            self.feature_names = ['userid', 'itemid', 'weekday', 'domain',
                                  'sales_chart', 'sales_rank', 'brand', 'price',
                                  'user_pos_1month_seq', 'user_neg_1month_seq',  # user history interaction
                                  'user_pos_2month_seq', 'user_neg_2month_seq',
                                  'user_pos_6month_seq', 'user_neg_6month_seq']
            self.domain2encoder_dict = {'Appliances': 0, 'Arts, Crafts & Sewing': 1, 'Automotive': 2, 'Books': 3,
                                        'CDs & Vinyl': 4, 'Cell Phones & Accessories': 5, 'Clothing, Shoes & Jewelry': 6,
                                        'Collectibles & Fine Art': 7, 'Electronics': 8, 'Gift Cards': 9,
                                        'Grocery & Gourmet Food': 10, 'Home & Business Services': 11, 'Home & Kitchen': 12,
                                        'Industrial & Scientific': 13, 'Kindle Store': 14, 'Magazine Subscriptions': 15,
                                        'Movies & TV': 16, 'Musical Instruments': 17, 'Office Products': 18,
                                        'Patio, Lawn & Garden': 19, 'Pet Supplies': 20, 'Sports & Outdoors': 21,
                                        'Tools & Home Improvement': 22, 'Toys & Games': 23, 'Video Games': 24}
            self.feature_dims, self.itemid_all = None, None
            self.merge_month = 6
            self.prepare2train_month = prepare2train_month
            self.preprocess_path = os.path.join(self.data_path, f'prepare2train_filter_{self.prepare2train_month}month.csv')
            self.label_name = 'label'
        elif dataset_name == 'aliccp':
            categorical_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207',
                                   '210', '216', '508', '509', '702', '853', '109_14', '110_14', '127_14', '150_14',
                                   '301']
            numerical_columns = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
            self.feature_names = categorical_columns + numerical_columns
            self.domain2encoder_dict = {str(item): item for item in range(self.sample_n_domain)}
            self.preprocess_path = os.path.join(self.data_path,
                                                f'thresh{self.downsample_freq_thresh}_ndomain{self.sample_n_domain}_mode{self.sample_mode}.csv')
            self.label_name = 'click'
        elif dataset_name == 'cloudtheme':
            self.feature_names = ['userid', 'itemid', 'domain', 'leaf_cate_id', 'cate_level1_id']
            self.domain2encoder_dict = {str(item): item for item in range(self.sample_n_domain)}
            self.preprocess_path = os.path.join(self.data_path,
                                                f'kcore{self.k_cores}_ndomain{self.sample_n_domain}_mode{self.sample_mode}_neg{negative_sampling_ratio}.csv')
            self.label_name = 'click'
            self.negative_sampling_ratio = negative_sampling_ratio

        self.one_hot_feature_names = [f for f in self.feature_names if 'seq' not in f]
        self.feature_dims, self.itemid_all = None, None
        self.is_aug = is_aug
        if self.is_aug:
            if aug_ratio > 0:
                self.aug_ratio = aug_ratio
            else:
                raise ValueError("aug_ratio must be greater than 0")
            self.preprocess_aug_path = self.preprocess_path.replace('.csv', f'_aug{aug_ratio}.csv')
        else:
            self.aug_ratio, self.preprocess_aug_path = None, None

    def update_config(self, config):
        """
        update dataset info in config
        """
        config.domain2encoder_dict = self.domain2encoder_dict
        config.preprocess_path = self.preprocess_path
        config.preprocess_aug_path = self.preprocess_aug_path

    def merge_metadata(self, df, add_features, k_cores):  # for amazon dataset
        def process_price(price_str):
            try:
                if not isinstance(price_str, str) or pd.isnull(price_str) or price_str == '':
                    return None
                cleaned_price = re.sub('[^\d.-]', '', price_str)
                if '-' in cleaned_price:
                    prices = cleaned_price.split('-')
                    price = np.mean([float(p) for p in prices])
                else:
                    price = float(cleaned_price)
                return np.ceil(price)
            except ValueError:
                return None

        def process_rank(sales_rank_str):
            if not isinstance(sales_rank_str, str):
                return None, None
            try:
                rank_part, chart_part = sales_rank_str.split(' in ')
                rank = int(rank_part.replace(',', ''))
                chart = chart_part.split(' (')[0]
                return rank, chart
            except ValueError:
                return None, None

        metadata_path = os.path.join(self.data_path, 'All_Amazon_Meta.json')

        # k-cores filter
        print('before k-cores filter: df shape = ', df.shape)
        df['user_count'] = df.groupby('userid')['userid'].transform('count')
        df['item_count'] = df.groupby('itemid')['itemid'].transform('count')
        df = df.loc[df.user_count >= k_cores]
        df = df.loc[df.item_count >= k_cores].copy()
        unique_items = set(df.itemid.unique())
        nunique_items = df.itemid.nunique()
        print(f'after k-cores filter: df shape = {df.shape}')
        print(f'user unique = {df.userid.nunique()}, item unique = {df.itemid.nunique()}')

        # read item metadata
        item_meta_df_path = os.path.join(self.data_path, f'item_meta_{self.k_cores}cores_{self.merge_month}month.csv')
        if os.path.exists(item_meta_df_path):
            item_meta_df = pd.read_csv(item_meta_df_path)
        else:
            item_meta_df = list()
            item_cnt = 0
            with open(metadata_path, 'rb') as f:
                tqdm_bar = tqdm(f, smoothing=0, mininterval=100.0)
                for line in tqdm_bar:
                    line = json.loads(line)
                    if line['asin'] not in unique_items:
                        continue
                    item_meta_df.append([line['asin'], line['price'], line['rank'], line['brand'], line['category']])

                    item_cnt += 1
                    if item_cnt % 1000 == 0:
                        tqdm_bar.set_description(f"Processed {item_cnt}/{nunique_items} items")

                    if item_cnt >= nunique_items:
                        break
            item_meta_df = pd.DataFrame(item_meta_df, columns=['itemid', 'price', 'salesRank', 'brand', 'category'])
            item_meta_df.to_csv(item_meta_df_path, index=False)
        print(f'item_meta_df shape is {item_meta_df.shape}')

        # process item meta data
        item_meta_df.replace('', None, inplace=True)
        item_meta_df['price'] = item_meta_df['price'].apply(process_price)
        item_meta_df['sales_rank'], item_meta_df['sales_chart'] = zip(*item_meta_df['salesRank'].apply(process_rank))
        item_meta_df['tags'] = item_meta_df['category'].apply(ast.literal_eval)
        item_meta_df['domain'] = item_meta_df['tags'].apply(lambda x: x[0] if isinstance(x, list)
                                                                              and len(x) > 0 else None)
        brand_counts = item_meta_df['brand'].value_counts()
        brands_to_replace = brand_counts[brand_counts < 10].index
        item_meta_df['brand'] = item_meta_df['brand'].apply(lambda x: None if x in brands_to_replace else x)

        # process label
        df['user_mean'] = df.groupby(by='userid')['rating'].transform('mean')
        df['label'] = 0
        df.loc[(df.rating > df.user_mean), 'label'] = 1

        # encoder itemid
        lbe = LabelEncoder()
        lbe.fit(list(unique_items))
        df['itemid'] = lbe.transform(df['itemid'].astype(str))
        item_meta_df['itemid'] = lbe.transform(item_meta_df['itemid'].astype(str))
        with open(os.path.join(self.data_path, 'itemid_encoder.pkl'), 'wb') as f:
            pickle.dump(lbe, f)

        # process user metadata
        start = time.time()
        df.sort_values('timestamp', inplace=True, ignore_index=True)
        pos_df, neg_df = df.loc[df.label == 1].copy(), df.loc[df.label == 0].copy()
        pos_user_meta_df = pos_df.groupby('userid').agg({
            'itemid': lambda x: list(x),
            'timestamp': lambda x: list(x)
        }).reset_index().rename(columns={'itemid': 'pos_item_seq', 'timestamp': 'pos_item_seq_timestamp'})
        neg_user_meta_df = neg_df.groupby('userid').agg({
            'itemid': lambda x: list(x),
            'timestamp': lambda x: list(x)
        }).reset_index().rename(columns={'itemid': 'neg_item_seq', 'timestamp': 'neg_item_seq_timestamp'})
        print(f'pos_user_meta_df shape = {pos_user_meta_df.shape}, neg_user_meta_df shape = {neg_user_meta_df.shape}')
        user_meta_df = pd.merge(pos_user_meta_df, neg_user_meta_df, on='userid', how='outer')
        user_meta_df['pos_item_seq'] = user_meta_df['pos_item_seq'].apply(lambda x: x if isinstance(x, list) else [])
        user_meta_df['pos_item_seq_timestamp'] = user_meta_df['pos_item_seq_timestamp'].apply(lambda x: x if isinstance(x, list) else [])
        user_meta_df['neg_item_seq'] = user_meta_df['neg_item_seq'].apply(lambda x: x if isinstance(x, list) else [])
        user_meta_df['neg_item_seq_timestamp'] = user_meta_df['neg_item_seq_timestamp'].apply(lambda x: x if isinstance(x, list) else [])
        end = time.time()
        print(f'user_meta_df shape = {user_meta_df.shape}, build time = {end - start:.2f}s')

        # merge user_meta_df to df
        start = time.time()
        df.sort_values('userid', inplace=True, ignore_index=True)
        user_meta_df.sort_values('userid', inplace=True, ignore_index=True)
        df['df2user_meta_df'] = user_meta_df['userid'].searchsorted(df['userid'], side='left')

        def get_user_items_seq(row, user_meta_df, delta_days, is_pos):
            user_meta_row = user_meta_df.iloc[row['df2user_meta_df']]
            start_time = row['timestamp'] - delta_days
            end_time = row['timestamp']
            item_seq = user_meta_row['pos_item_seq'] if is_pos else user_meta_row['neg_item_seq']
            item_seq_timestamp = user_meta_row['pos_item_seq_timestamp'] if is_pos \
                else user_meta_row['neg_item_seq_timestamp']
            selected_items = [item for item, timestamp in zip(item_seq, item_seq_timestamp) if
                              start_time <= timestamp < end_time]
            return selected_items

        m = 6
        days_n = 30*m
        delta_days = int(timedelta(days=days_n - 1).total_seconds())
        df[f'user_pos_{m}month_seq'] = df.apply(get_user_items_seq, axis=1,
                                                user_meta_df=user_meta_df, delta_days=delta_days, is_pos=True)
        df[f'user_neg_{m}month_seq'] = df.apply(get_user_items_seq, axis=1,
                                                user_meta_df=user_meta_df, delta_days=delta_days, is_pos=False)
        print(f'finish getting df.user_pos/neg_{m}month_seq')
        end = time.time()
        print(f'df shape = {df.shape}, get df.item_seq time = {end - start:.2f}s')

        df = df.merge(item_meta_df, on='itemid', how='left')
        print('finish merge item meta data to df')

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['weekday'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour  # hour都是0

        return df

    def filter_dataframe_by_threshold(self, df_paths, thresh, n_domain, sample_mode):
        """
        for aliccp/cloudtheme dataset,
        retain users and items with frequency greater than 'thresh',
        and sample 'n_domain' domains based on 'sample_mode'
        """
        with open(f"{self.preprocess_path.split(',')[0]}.log", 'w') as log_file:
            if isinstance(df_paths, pd.DataFrame):
                df = reduce_mem(df_paths)
            else:
                df_num = len(df_paths)
                train_tags = [0, 1, 2]
                dfs, df_row_nums = [], []
                for i in range(df_num):
                    dfs.append(reduce_mem(pd.read_csv(df_paths[i])))
                    dfs[i]['train_tag'] = train_tags[i]  # add a tag column to distinguish train, val, test
                    df_row_nums.append(dfs[i].shape[0])
                df = pd.concat(dfs, ignore_index=True)

            import sys
            sys.stdout = log_file

            print('Columns:', df.columns)
            if self.dataset_name == 'aliccp':
                print('Train_tag:', train_tags[:df_num])
                print(f"Concat {df_num} dataframes to filter, original row num: {df_row_nums}")

            # calculate the frequency of users and items
            user_counts = df['userid'].value_counts()
            item_counts = df['itemid'].value_counts()

            if thresh > 1:
                # filter out users and items with frequency less than thresh
                valid_users = user_counts[user_counts >= thresh].index
                valid_items = item_counts[item_counts >= thresh].index
                valid_mask = df['userid'].isin(valid_users) & df['itemid'].isin(valid_items)

                print("Before filter user and item:", df.shape[0])
                filtered_df = df[valid_mask]  # after filtering by user and item frequency
            else:
                print("Before filter user and item:", df.shape[0])
                filtered_df = df
            print("After filter user and item:", filtered_df.shape[0])

            # Filtering based on userid and itemid counts within each domain
            print("Before filter domain:", filtered_df["domain"].value_counts())
            filtered_df = filtered_df.groupby('domain').filter(
                lambda x: (x['userid'].nunique() >= thresh * 5) & (x['itemid'].nunique() >= thresh * 5))
            sort_by_count = filtered_df["domain"].value_counts().sort_values(ascending=False)
            print("After filter domain:", sort_by_count)
            print("domain counts describe:",
                  sort_by_count.describe(percentiles=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

            if sample_mode == "nlargest":
                selected_domains = sort_by_count.nlargest(n_domain).index
            elif sample_mode == "random":
                # Randomly selecting n_domain domains
                remaining_domains = set(filtered_df['domain'].unique())
                selected_domains = random.sample(remaining_domains, min(n_domain, len(remaining_domains)))
            elif sample_mode == "interval":
                # Sort domains based on count and select n evenly spaced domains
                sorted_domains = sort_by_count.index
                step = max(1, len(sorted_domains) // n_domain)
                selected_domains = sorted_domains[::step][:n_domain]
            elif sample_mode == "weighted":
                # Calculate weights based on log-transformed domain count
                domain_counts = filtered_df["domain"].value_counts()
                mid = domain_counts.median()
                domain_counts_f = (domain_counts + 0.2 * mid ** 2 / domain_counts) ** 0.8
                weights = domain_counts_f / domain_counts_f.sum()
                print("weights:", weights)
                selected_domains = np.random.choice(domain_counts.index, n_domain, p=weights, replace=False)
            elif sample_mode == "interval_random":  # stratified sampling
                # Sort domains based on count and select n domains from each interval
                sorted_domains = sort_by_count.index
                large_domains = sorted_domains[:int(0.05 * len(sorted_domains))]
                small_domains = sorted_domains[int(0.05 * len(sorted_domains)):]

                selected_domains = []
                large_cnt = max(5, int(n_domain * 0.15))
                for tmp_n_domains, tmp_sorted_domains in zip([large_cnt, n_domain - large_cnt], [large_domains, small_domains]):
                    step = max(1, len(tmp_sorted_domains) // tmp_n_domains)
                    selected_domains.extend(tmp_sorted_domains[::step][:tmp_n_domains])
            else:
                raise ValueError("Invalid sample_mode")

            print("sample_mode:", sample_mode)
            print("selected_domains:", selected_domains)
            filtered_df = filtered_df[filtered_df['domain'].isin(selected_domains)]
            print("After select domain with sample_mode:")
            print("After final sample domain 1:", filtered_df["domain"].value_counts())

            # Mapping domains to continuous IDs
            domain_id_mapping = {domain: i for i, domain in enumerate(selected_domains)}
            domain_id_mapping_str = {str(domain): i for i, domain in enumerate(selected_domains)}
            inverse_domain_id_mapping = {i: domain for domain, i in domain_id_mapping.items()}
            self.domain2encoder_dict = domain_id_mapping_str
            filtered_df['domain'] = filtered_df['domain'].map(domain_id_mapping)

            # some user and item may be missing after sampling, so re-encoding
            print("Re-encoding userid and itemid after domain sampling")
            print(f"Before re-encoding, userid max: {filtered_df['userid'].max()}, "
                  f"itemid max: {filtered_df['itemid'].max()}")
            if self.dataset_name == 'aliccp':
                reencode_cols = ['userid', 'itemid']
            elif self.dataset_name == 'cloudtheme':
                reencode_cols = [col for col in self.feature_names if col!='domain']
            for fea in reencode_cols:
                lbe = LabelEncoder()
                filtered_df[fea] = lbe.fit_transform(filtered_df[fea])
            print(f"After re-encoding, userid max: {filtered_df['userid'].max()}, "
                  f"itemid max: {filtered_df['itemid'].max()}")

            print("After final sample domain 2:", filtered_df["domain"].value_counts(),
                  "len", len(filtered_df["domain"]))
            sys.stdout = sys.__stdout__
        print("After final sample domain 3:", filtered_df["domain"].value_counts(),
              "len", len(filtered_df["domain"]))

        return filtered_df, domain_id_mapping, inverse_domain_id_mapping

    def make_augmentation(self):
        """
        Popularity-based Counterfactual Augmentation.
        Based on Corollary 1 described in the paper.
        """
        if os.path.exists(self.preprocess_aug_path):
            print(f'augmentation data {self.preprocess_aug_path} already prepared')
        else:
            with open(f"{self.preprocess_aug_path}.log", 'w') as log_file:
                data = pd.read_csv(self.preprocess_path)
                aug_len = int(data.shape[0] * self.aug_ratio)

                import sys
                sys.stdout = log_file

                if self.dataset_name == 'amazon':
                    positive_smoothing, total_smoothing = 1, 2
                elif self.dataset_name == 'aliccp':
                    positive_smoothing, total_smoothing = 1, 2
                elif self.dataset_name == 'cloudtheme':
                    positive_smoothing, total_smoothing = 1, 2
                print('dataset_name:', self.dataset_name)
                print(f'positive_smoothing: {positive_smoothing}, total_smoothing: {total_smoothing}')
                if self.dataset_name == 'cloudtheme':
                    item_popularity = data.groupby('itemid').agg({'clk_cnt': ['count', 'sum']})
                else:
                    item_popularity = data.groupby('itemid').agg({self.label_name: ['count', 'sum']})
                item_popularity.columns = ['total_count', 'positive_count']
                item_popularity['popularity'] = (item_popularity['positive_count'] + positive_smoothing) / (
                            item_popularity['total_count'] + total_smoothing)
                print('item_popularity')
                print(item_popularity.describe())

                domain_counts = data['domain'].value_counts()
                data['is_augmented'] = False

                if self.dataset_name == 'amazon':
                    cold_item_threshold = 4
                    # select cold items based on exposure
                    cold_items = item_popularity[item_popularity['total_count'] <= cold_item_threshold].index.to_numpy()

                    small_domain_threshold = int(data.shape[0] * 0.02)  # minority domain threshold
                    large_domains = domain_counts[domain_counts > 1.5 * small_domain_threshold].index
                    small_domains = domain_counts[domain_counts <= small_domain_threshold].index

                    print(f'choose cold items with total_count <= cold_item_threshold')
                    print(f'small_domain_threshold: {small_domain_threshold}, '
                          f'cold_item_threshold: {cold_item_threshold}')
                elif self.dataset_name == 'aliccp':
                    popularity_threshold = 0.05
                    # select cold items based on popularity
                    cold_items = item_popularity[item_popularity['popularity'] < popularity_threshold].index.to_numpy()

                    small_domain_threshold = int(data.shape[0] * 0.015)  # minority domain threshold
                    large_domains = domain_counts[domain_counts > small_domain_threshold].index
                    small_domains = domain_counts[domain_counts <= small_domain_threshold].index

                    print(f'choose cold items with popularity < popularity_threshold')
                    print(f'small_domain_threshold: {small_domain_threshold}, '
                          f'popularity_threshold: {popularity_threshold}')
                elif self.dataset_name == 'cloudtheme':
                    popularity_threshold = 0.2
                    # select cold items based on popularity
                    cold_items = item_popularity[item_popularity['popularity'] < popularity_threshold].index.to_numpy()

                    small_domain_threshold = int(data.shape[0] * 0.015)  # minority domain threshold
                    large_domains = domain_counts[domain_counts > 1.5 * small_domain_threshold].index
                    small_domains = domain_counts[domain_counts <= small_domain_threshold].index

                    print(f'choose cold items with popularity < popularity_threshold')
                    print(f'small_domain_threshold: {small_domain_threshold}, '
                          f'popularity_threshold: {popularity_threshold}')

                print(f'total item num: {item_popularity.shape[0]}, cold item num: {len(cold_items)}')

                cold_items_in_large_domain = data[data['itemid'].isin(cold_items)
                                                  & data['domain'].isin(large_domains) & data[self.label_name] == 1]
                print(f'total sample num: {data.shape[0]}, '
                      f'augmentation from {cold_items_in_large_domain.shape[0]} samples, '
                      f'ratio: {cold_items_in_large_domain.shape[0]*100./data.shape[0]:.2f}%')

                # item sampling weights: less popular items have a higher chance of being selected for augmentation.
                tmp_item_weights = 1/item_popularity.loc[cold_items_in_large_domain['itemid'], 'popularity']
                item_weights = (tmp_item_weights / tmp_item_weights.sum()).tolist()
                augmented_samples = cold_items_in_large_domain.sample(n=aug_len, replace=True, weights=item_weights)

                # domain sampling weights: minority domains are more likely to receive augmented data.
                each_domain_num = (domain_counts.loc[small_domains].sum() + aug_len) / len(small_domains)
                weights = each_domain_num - domain_counts.loc[small_domains]
                weights.loc[weights < 100] = 100
                weights = np.exp(weights / weights.quantile(0.3))
                domain_weights = weights / weights.sum()
                augmented_samples['domain'] = np.random.choice(small_domains, size=aug_len, p=domain_weights)
                augmented_samples['is_augmented'] = True
                print(f'augmented samples num: {augmented_samples.shape[0]}')

                aug_domain_counts = augmented_samples['domain'].value_counts()
                ori_aug_domain_count = pd.concat([domain_counts, aug_domain_counts], axis=1)
                print('After augmentation, domain counts:')
                print(ori_aug_domain_count)

                data_augmented = pd.concat([data, augmented_samples])
                data_augmented.to_csv(self.preprocess_aug_path, index=False)

                sys.stdout = sys.__stdout__

            print(f'finish preprocessing augmentation data {self.preprocess_aug_path}')

    def main(self):
        if os.path.exists(self.preprocess_path):
            print(f'{self.preprocess_path} already prepared')
        else:
            if self.dataset_name == 'amazon':
                mergemeta_path = os.path.join(self.data_path, f'mergemeta_{self.k_cores}cores_{self.merge_month}month.csv')
                if os.path.exists(mergemeta_path):
                    df = pd.read_csv(mergemeta_path)
                else:
                    csv_path = os.path.join(self.data_path, f'all_csv_files_{self.merge_month}month.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path, engine='c', low_memory=False, on_bad_lines='skip')
                    else:
                        rating_csv_columns = ['itemid', 'userid', 'rating', 'timestamp']
                        df = pd.DataFrame(columns=rating_csv_columns)

                        # 存最近merge_month的交互记录
                        days_n = 30 * self.merge_month + self.merge_month // 2
                        end_date = int(datetime(2018, 8, 15).timestamp())  # df_total['timestamp'].max()
                        start_date = end_date - int(timedelta(days=days_n).total_seconds())

                        # 分块读取和处理 CSV 文件
                        chunksize = 5e7
                        for chunk in pd.read_csv(os.path.join(self.data_path, 'all_csv_files.csv'),
                                                 chunksize=chunksize, header=None, names=rating_csv_columns, engine='c',
                                                 low_memory=False, on_bad_lines='skip'):
                            filtered_chunk = chunk.loc[(chunk['timestamp'] >= start_date) & (chunk['timestamp'] < end_date)]
                            df = pd.concat([df, filtered_chunk], ignore_index=True)

                        df.to_csv(csv_path, index=False)
                    print(f'df total shape = {df.shape}')

                    # 合并product的meta特征
                    df = self.merge_metadata(df, add_features=self.feature_names[2:], k_cores=self.k_cores)
                    df.to_csv(mergemeta_path, index=False)

                print('finish loading data. start preprocessing')

                if self.prepare2train_month < self.merge_month:
                    # 取部分月的数据
                    print(f'filter {self.prepare2train_month}-month from {self.merge_month}-month merge_meta data')
                    end_date = df['timestamp'].max()
                    days_n = 30 * self.prepare2train_month + self.prepare2train_month // 2
                    start_date = end_date - int(timedelta(days=days_n-1).total_seconds())
                    df = df.loc[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()

                # 稠密特征离散化
                df['sales_rank'] = df['sales_rank'].fillna(df['sales_rank'].quantile()).astype(int)  # sales_rank
                sales_rank_bins = [0] + list(np.exp2(np.arange(2, 21, 2)).astype(int)) + [np.inf]
                df['sales_rank'] = pd.cut(df['sales_rank'], bins=sales_rank_bins, labels=False)

                df['price'] = df['price'].fillna(df['price'].quantile()).astype(int)  # sales_rank
                price_bins = [-1] + list(np.exp2(np.arange(1, 13, 1.2)).astype(int)) + [np.inf]
                df['price'] = pd.cut(df['price'], bins=price_bins, labels=False)
                df['timestamp'] = df['timestamp'].astype(int)

                # 定长特征数据数字化，itemid already encoded
                encoder_feature_names = [fea for fea in self.one_hot_feature_names if (fea!='itemid') and (fea!='domain')]
                df[encoder_feature_names].fillna('-1', inplace=True)
                for fea in encoder_feature_names:
                    lbe = LabelEncoder()
                    df[fea] = lbe.fit_transform(df[fea].astype(str))

                df = df.loc[df['domain'].isin(self.domains)] if len(self.domains) > 0 else df
                df = df.dropna(subset=['domain'])
                df['domain'] = df['domain'].map(self.domain2encoder_dict)

                data = df[self.feature_names+['label']+['timestamp']]  # timestamp是后续划分训练测试集需要
                data.to_csv(self.preprocess_path, index=False)
                print(f'finish preprocessing {self.preprocess_path}')
            elif self.dataset_name == 'aliccp':
                discrete_paths = (os.path.join(self.data_path, f"ali_ccp_train_discrete_{self.discrete_method}.csv"),
                                  os.path.join(self.data_path, f"ali_ccp_val_discrete_{self.discrete_method}.csv"),
                                  os.path.join(self.data_path, f"ali_ccp_test_discrete_{self.discrete_method}.csv"))

                def discrete(KBinsDiscretizer_paths):
                    print("Discretize continuous features, fit and transform on train, transform on val and test")
                    print(discrete_paths)
                    # train_path, val_path, test_path
                    train_val_test_path = (os.path.join(self.data_path, 'ali_ccp_train.csv'),
                                           os.path.join(self.data_path, 'ali_ccp_val.csv'),
                                           os.path.join(self.data_path, 'ali_ccp_test.csv'))
                    if not all([os.path.exists(path) for path in train_val_test_path]):
                        print("Train, val, test data not prepared, preprocess_ali_ccp.py will be executed")
                        import subprocess
                        preprocess_ali_ccp_path = os.path.join(self.data_path, 'preprocess_ali_ccp.py')
                        raise ValueError("Please run preprocess_ali_ccp.py first")
                        print("Finish preprocess_ali_ccp.py")
                    else:
                        print("Train, val, test data already prepared")
                    train_val_test_df = (pd.read_csv(train_val_test_path[0]),
                                         pd.read_csv(train_val_test_path[1]),
                                         pd.read_csv(train_val_test_path[2]))
                    print("train_val_test_df:", [df.shape for df in train_val_test_df])

                    from sklearn.preprocessing import KBinsDiscretizer
                    columns_to_discretize = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
                    print("columns_to_discretize:", columns_to_discretize)

                    # use KBinsDiscretizer to discretize
                    for column in tqdm(columns_to_discretize, mininterval=5):
                        discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal',
                                                       strategy=self.discrete_method,
                                                       subsample=int(2e5) if self.discrete_method == 'quantile' else None)
                        discretizer.fit(train_val_test_df[0][[column]])  # fit only on train data, avoid data leakage
                        for i in range(3):
                            train_val_test_df[i][column] = discretizer.transform(train_val_test_df[i][[column]]).astype(int)

                    for i in range(3):
                        train_val_test_df[i].rename(columns={'101': 'userid', '205': 'itemid', '206': 'domain'},
                                                    inplace=True)
                        train_val_test_df[i].to_csv(discrete_paths[i], index=False)
                    print("Discretization done")

                if not all([os.path.exists(path) for path in discrete_paths]):
                    discrete(discrete_paths)
                else:
                    print("Discrete data already prepared")

                df, domain_id_mapping, inverse_domain_id_mapping = self.filter_dataframe_by_threshold(discrete_paths,
                                                                                                      self.downsample_freq_thresh,
                                                                                                      self.sample_n_domain,
                                                                                                      self.sample_mode)
                df.to_csv(self.preprocess_path, index=False)
            elif self.dataset_name == 'cloudtheme':
                df = pd.read_csv(os.path.join(self.data_path, 'theme_click_log.csv'),
                                 engine='c', low_memory=False, on_bad_lines='skip')
                df.rename(columns={'user_id': 'userid', 'item_id': 'itemid', 'theme_id': 'domain'}, inplace=True)

                # first encode feature
                self.one_hot_feature_names = self.feature_names
                for fea in self.one_hot_feature_names:
                    lbe = LabelEncoder()
                    df[fea] = lbe.fit_transform(df[fea].astype(str))

                df, domain_id_mapping, inverse_domain_id_mapping = self.filter_dataframe_by_threshold(df,
                                                                                                      self.k_cores,
                                                                                                      self.sample_n_domain,
                                                                                                      self.sample_mode)

                # sort by time and add train_tag
                df = df.sort_values(by='reach_time')
                index_80, index_90 = int(len(df) * 0.8), int(len(df) * 0.9)
                df['train_tag'] = 0
                df.iloc[index_80:index_90, df.columns.get_loc('train_tag')] = 1
                df.iloc[index_90:, df.columns.get_loc('train_tag')] = 2
                train_data, val_data, test_data = df.iloc[:index_80], df.iloc[index_80:index_90], df.iloc[index_90:]

                def generate_negative_samples(sample_item_data, user_replace_pool, n_neg_samples, all_positive_pair,
                                              train_tag):
                    # smoothing sample probability
                    smoothed_clk_cnt = np.log1p(sample_item_data['clk_cnt'])  # log1p = log(x+1)
                    negative_samples = sample_item_data.sample(n=n_neg_samples, replace=True, weights=smoothed_clk_cnt)

                    # randomly replace userid
                    replace_user_ids = user_replace_pool['userid'].sample(n=n_neg_samples, replace=True).values
                    negative_samples['userid'] = replace_user_ids

                    merged = pd.merge(negative_samples, all_positive_pair[['userid', 'itemid']],
                                      on=['userid', 'itemid'], how='left', indicator=True)
                    new_negative_samples = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

                    new_negative_samples['train_tag'] = train_tag
                    new_negative_samples['click'] = 0
                    new_negative_samples['clk_cnt'] = 0
                    print(f'*{train_tag}* negative samples: before drop duplicates-{negative_samples.shape[0]}, '
                          f'after drop duplicates-{new_negative_samples.shape[0]}')

                    return new_negative_samples

                # negative sampling
                print('Negative sampling ratio:', self.negative_sampling_ratio)
                # avoid data leakage
                neg_train_samples = generate_negative_samples(sample_item_data=train_data,
                                                              user_replace_pool=train_data,
                                                              n_neg_samples=int(len(train_data) * self.negative_sampling_ratio),
                                                              all_positive_pair=df, train_tag=0)
                neg_val_samples = generate_negative_samples(sample_item_data=df.iloc[:index_90],  # train+val
                                                            user_replace_pool=val_data,
                                                            n_neg_samples=int(len(val_data) * self.negative_sampling_ratio),
                                                            all_positive_pair=df, train_tag=1)
                neg_test_samples = generate_negative_samples(sample_item_data=df,  # train+val+test
                                                             user_replace_pool=test_data,
                                                             n_neg_samples=int(len(test_data) * self.negative_sampling_ratio),
                                                             all_positive_pair=df, train_tag=2)

                df['click'] = 1
                save_cols = self.feature_names + ['click', 'train_tag', 'clk_cnt']
                all_data = pd.concat([df[save_cols], neg_train_samples[save_cols], neg_val_samples[save_cols],
                                      neg_test_samples[save_cols]], ignore_index=True)
                print('all_data shape:', all_data.shape,
                      ' positive ratio:', len(df) / len(all_data))

                all_data.to_csv(self.preprocess_path, index=False)

        if self.is_aug:
            self.make_augmentation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_cores', default=3)
    parser.add_argument('--seed', type=int, default=2000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    DataPreprocessing('dataset/aliccp', 'aliccp', domains=[], k_cores=args.k_cores, is_aug=True, aug_ratio=0.1).main()