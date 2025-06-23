import math
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datasets.python_splitters import python_stratified_split
from collections import Counter
import torch
import pickle
import os
from module import config


def construct_dict(filename, user_set, item_set):
    dicts = []
    with open(filename) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                items = [int(i) for i in line[1:]]
                uid = int(line[0])
                for iid in items:
                    dicts.append({'userID': uid, 'itemID': iid, 'rating': 1})
                    item_set.add(iid)
                user_set.add(uid)
    return pd.DataFrame.from_records(dicts), user_set, item_set


class ImplicitCF(object):
    """Data processing class for GCN models which use implicit feedback.

    Initialize train and test set, create normalized adjacency matrix and sample data for training epochs.

    """
    def __init__(
        self,
        base_model,
        dataset,
        with_validation=False
    ):
        self.user_idx = None
        self.item_idx = None
        self.adj_dir = 'tmp/'
        self.col_user = 'userID'
        self.col_item = 'itemID'
        self.col_rating = 'rating'
        self.base_model = base_model
        self.dataset_type = dataset

        print('Using validation set: ', with_validation)

        if dataset == 'ml-1m':
            filename = 'data/ml-1m/ratings.dat'
            header = ['userID', 'itemID', 'rating', 'timestamp']
            dtypes = {h: np.int32 for h in header}
            df = pd.read_csv(
                filename, sep='::', header=0, names=header, engine='python', dtype=dtypes
            )

        elif dataset == 'yelp' or dataset == 'gowalla':
            train_file = 'data/{}/train.txt'.format(dataset)
            test_file = 'data/{}/test.txt'.format(dataset)

            train, user_set, item_set = construct_dict(train_file, set(), set())
            test, user_set, item_set = construct_dict(test_file, user_set, item_set)
            df = pd.concat([train, test])

        elif dataset == 'ml-25m':
            filename = 'data/ml-25m/ratings.csv'
            header = ['userID', 'itemID', 'rating']
            dtypes = {h: np.int32 for h in header}
            df = pd.read_csv(
                filename, sep=',', header=0, names=header, engine='python', dtype=dtypes
            )
        else:
            raise ValueError('Invalid dataset!')

        train, test = python_stratified_split(df, ratio=[0.75, 0.25])
        if with_validation:
            train, validation = python_stratified_split(train, ratio=[2/3, 1/3])
            self.train, self.validation, self.test = self._data_processing(train, test, validation=validation)
            self.positive_validation_pairs = set(
                (pair[0], pair[1]) for pair in self.validation[['userID', 'itemID']].values
            )
            self.validation_size = len(self.validation)
        else:
            self.train, self.test = self._data_processing(train, test)

        assert min(self.train['userID'].values) == 0 and min(self.train['itemID'].values) == 0

        # setting up R and interacted_status
        self._init_train_data()

        self.positive_test_pairs = set(
            (pair[0], pair[1]) for pair in self.test[['userID', 'itemID']].values
        )

        self.train_size = len(self.train)
        self.test_size = len(self.test)

        self.user_freq = Counter(self.train['userID'])
        self.max_user_freq = max(self.user_freq.values())
        self.min_user_freq = min(self.user_freq.values())

        self.item_freq = Counter(self.train['itemID'])
        self.max_item_freq = max(self.item_freq.values())
        self.min_item_freq = min(self.item_freq.values())

        self.user_vocab = list(set(self.user_idx['userID_idx'].values))
        self.item_vocab = list(set(self.item_idx['itemID_idx'].values))

        if not with_validation:
            self.y_true = self.get_y_true(self.positive_test_pairs, False)
        else:
            self.y_true = self.get_y_true(self.positive_validation_pairs, True)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _data_processing(self, train, test, validation=None):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            validation (pandas.DataFrame): Validation data
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                test can be None, if so, we only process the training data.

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.

        """
        if validation is None:
            df = pd.concat([train, test])
        else:
            df = pd.concat([train, validation, test])

        # replace IDs with indices
        if self.user_idx is None:
            user_idx = df[[self.col_user]].drop_duplicates().reindex()
            user_idx[self.col_user + "_idx"] = np.arange(len(user_idx))
            self.n_users = len(user_idx)
            self.user_idx = user_idx

        if self.item_idx is None:
            item_idx = df[[self.col_item]].drop_duplicates()
            item_idx[self.col_item + "_idx"] = np.arange(len(item_idx))
            self.n_items = len(item_idx)
            self.item_idx = item_idx

        if validation is None:
            return self._reindex(train),  self._reindex(test)
        else:
            return self._reindex(train), self._reindex(validation), self._reindex(test)

    def get_freq(self, mode, entity):
        if mode == 'user':
            if entity in self.user_freq:
                freq_diff = self.max_user_freq - self.min_user_freq
                return (self.user_freq[entity] - self.min_user_freq) / freq_diff
            else:
                # does not exist in the training set
                return 0
        else:
            assert mode == 'item'
            if entity in self.item_freq:
                freq_diff = self.max_item_freq - self.min_item_freq

                return (self.item_freq[entity] - self.min_item_freq) / freq_diff
            else:
                return 0

    def get_y_true(self, positive_pairs, with_validation):
        y_trues = []
        folder = 'tmp/y_trues/{}/'.format(self.dataset_type)
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            for chunk in range(config.CHUNK_NUM):
                path = folder + 'y_true_{}_{}_{}_{}.pkl'.format(self.dataset_type, config.SEED, with_validation, chunk)
                with open(path, 'rb') as f:
                    y_trues.append(pickle.load(f))

        except Exception as e:
            print(e)
            print('Presaved y_true loading failed. Creating y_true...')
            chunk_size = math.ceil(self.n_users / config.CHUNK_NUM)
            folder = 'tmp/y_trues/{}/'.format(self.dataset_type)
            total_users = 0
            for chunk in range(config.CHUNK_NUM):
                start_ind = chunk * chunk_size
                end_ind = min(self.n_users, (chunk + 1) * chunk_size)
                users_in_chunk = set(self.user_vocab[start_ind: end_ind])
                path = folder + 'y_true_{}_{}_{}_{}.pkl'.format(self.dataset_type, config.SEED, with_validation, chunk)
                with open(path, 'wb') as f:
                    y_true_chunk = np.zeros((len(users_in_chunk), len(self.item_vocab)), dtype=np.int32)
                    for user_id, item_id in positive_pairs:
                        if user_id in users_in_chunk:
                            user_pos = user_id % chunk_size
                            y_true_chunk[user_pos][item_id] = 1
                    y_trues.append(y_true_chunk)
                    pickle.dump(y_trues[-1], f, protocol=pickle.HIGHEST_PROTOCOL)
                total_users += len(users_in_chunk)
            assert total_users == self.n_users
            print('y_true saved!')
        return y_trues

    def get_y_true_by_user(self, user_id):
        chunk_size = math.ceil(self.n_users / config.CHUNK_NUM)
        chunk = math.floor(user_id / chunk_size)
        goal_chunk = self.y_true[chunk]
        chunk_size = math.ceil(self.n_users / config.CHUNK_NUM)
        user_pos = user_id % chunk_size
        return goal_chunk[user_pos]

    def _reindex(self, df):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating).

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.

        """
        if df is None:
            return None

        df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        df = df[df[self.col_rating] > 0]

        df_reindex = df[
            [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        ]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df_reindex

    def _init_train_data(self):
        """Record items interated with each user in a dataframe self.interact_status, and create adjacency
        matrix self.R.

        """
        self.interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R[self.train[self.col_user], self.train[self.col_item]] = 1.0

    def get_norm_adj_mat(self):
        """Load normalized adjacency matrix if it exists, otherwise create (and save) it.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        return self.create_norm_adj_mat()

    def create_norm_adj_mat(self):
        """Create normalized adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[: self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, : self.n_users] = R.T
        if self.base_model == 'ngcf':
            adj_mat = adj_mat.todok() + sp.eye(adj_mat.shape[0])
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat).tocoo()
            return norm_adj_mat.tocsr()
        elif self.base_model == 'lightgcn':
            adj_mat = adj_mat.todok()
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_mat.dot(d_mat_inv)
            return norm_adj_mat.tocsr()

    def train_loader(self, batch_size):
        """Sample train data every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size (int): Batch size of users.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray:
            - Sampled users.
            - Sampled positive items.
            - Sampled negative items.
        """
        def sample_neg(x):
            while True:
                neg_id = random.randint(0, self.n_items - 1)
                if neg_id not in x:
                    return neg_id

        indices = range(len(self.interact_status))
        if len(indices) < batch_size:
            selected_indices = [random.choice(indices) for _ in range(batch_size)]
        else:
            selected_indices = random.sample(indices, batch_size)
        interact = self.interact_status.iloc[selected_indices]

        pos_items = interact[self.col_item + "_interacted"].apply(
            lambda x: random.choice(list(x))
        )
        neg_items = interact[self.col_item + "_interacted"].apply(
            lambda x: sample_neg(x)
        )
        users = np.array(interact[self.col_user])
        pos_items = np.array(pos_items)
        neg_items = np.array(neg_items)
        users = torch.tensor(users).long().to(config.device)
        pos_items = torch.tensor(pos_items).long().to(config.device)
        neg_items = torch.tensor(neg_items).long().to(config.device)

        return users, pos_items, neg_items
