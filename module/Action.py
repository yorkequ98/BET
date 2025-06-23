from module import config
import numpy as np
import torch
from module.Entity import Entity


class Action:
    def __init__(self, dataset, alpha, ints, distribution, user_weight, pvals, sort_sizes):
        self.buckets = []
        for _ in range(config.MIN_EMB_SIZE, config.MAX_EMB_SIZE + 1):
            self.buckets.append([[], []])

        # store the number of users in each bucket
        self.num_user_in_bucket = [0] * len(self.buckets)
        # store the number of items in each bucket
        self.num_item_in_bucket = [0] * len(self.buckets)

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        num_entities = dataset.n_users + dataset.n_items
        self.binary_masks = np.zeros((num_entities, len(self.buckets)), dtype=np.int32)
        self.array_form = np.zeros(num_entities, dtype=np.int32)

        self.timestamp = None

        self.metric = None

        self.alphas = alpha

        self.ints = ints
        self.pvals = pvals
        self.distribution = distribution
        self.user_weight = user_weight
        self.sort_sizes = sort_sizes

        labelled_users, labelled_items = self.get_sorted_triples(dataset, sort_sizes)
        user_ints, item_ints = self.ints

        count = 0
        for _, user_id, user_freq in labelled_users:
            integer = user_ints[count]
            entity = Entity(user_id, 'user', user_freq)
            self.add_entity(integer, entity)
            count += 1

        count = 0
        for _, item_id, item_freq in labelled_items:
            integer = item_ints[count]
            entity = Entity(item_id, 'item', item_id)
            self.add_entity(integer, entity)
            count += 1

    def get_sorted_triples(self, dataset, sorted_triples):
        labelled_users = []
        for ID in dataset.user_vocab:
            freq = dataset.get_freq('user', ID)
            labelled_users.append(('user', ID, freq))
        if sorted_triples:
            labelled_users.sort(key=lambda triple: triple[2], reverse=True)

        labelled_items = []
        for ID in dataset.item_vocab:
            freq = dataset.get_freq('item', ID)
            labelled_items.append(('item', ID, freq))
        if sorted_triples:
            labelled_items.sort(key=lambda triple: triple[2], reverse=True)

        return labelled_users, labelled_items

    def add_entity(self, size, entity):
        if entity.type == 'user':
            self.buckets[size - 1][0].append(entity)
            # update the max number of users
            self.num_user_in_bucket[size - 1] += 1
            index = entity.id
        else:
            # add an item
            self.buckets[size - 1][1].append(entity)
            # update the max number of items
            self.num_item_in_bucket[size - 1] += 1
            index = entity.id + self.n_users
        self.array_form[index] = size
        arr = np.zeros(config.MAX_EMB_SIZE - config.MIN_EMB_SIZE + 1)
        arr[:size] = 1
        self.binary_masks[index] = arr

    def get_array_form(self):
        user_sizes = self.array_form[:self.n_users]
        item_sizes = self.array_form[self.n_users:]
        assert len(user_sizes) == self.n_users
        assert len(item_sizes) == self.n_items
        return user_sizes, item_sizes

    def get_set_form(self):
        hidden_size = 16
        num_sets = len(self.buckets)
        max_user_len = max(self.num_user_in_bucket)
        max_item_len = max(self.num_item_in_bucket)
        # size of the largest set
        max_len = max_user_len + max_item_len
        # create a binary mask
        mask = np.zeros((num_sets, max_len, hidden_size), dtype=np.int32)
        user_action_data = np.zeros((num_sets, max_user_len))
        item_action_data = np.zeros((num_sets, max_item_len))
        for i in range(num_sets):
            # users
            mask[i][:self.num_user_in_bucket[i]] = 1
            users_in_bucket = [self.buckets[i][0][j].get_input_form() for j in range(len(self.buckets[i][0]))]
            assert len(self.buckets[i][0]) == self.num_user_in_bucket[i]
            user_action_data[i][:self.num_user_in_bucket[i]] = users_in_bucket[:max_user_len]
            assert len(user_action_data[i][:self.num_user_in_bucket[i]]) == len(users_in_bucket)

            # items
            mask[i][max_user_len: max_user_len + self.num_item_in_bucket[i]] = 1
            items_in_bucket = [self.buckets[i][1][j].get_input_form() for j in range(len(self.buckets[i][1]))]
            assert len(self.buckets[i][1]) == self.num_item_in_bucket[i]
            item_action_data[i][:self.num_item_in_bucket[i]] = items_in_bucket[:max_item_len]
            assert len(item_action_data[i][:self.num_item_in_bucket[i]]) == len(items_in_bucket)

        user_action_data = torch.tensor(user_action_data, device=config.device)
        item_action_data = torch.tensor(item_action_data, device=config.device)
        mask = torch.tensor(mask, device=config.device)
        num_nonzeros = np.array([self.num_user_in_bucket[i] + self.num_item_in_bucket[i] for i in range(num_sets)])
        num_nonzeros = np.maximum(num_nonzeros, np.ones_like(num_nonzeros))
        num_nonzeros = torch.tensor(num_nonzeros, device=config.device)
        return user_action_data, item_action_data, mask, num_nonzeros

    def __eq__(self, other):
        if other is None:
            return False
        return self.timestamp == other.timestamp

    def __str__(self):
        user_sizes, item_sizes = self.ints
        num_entities = self.n_users + self.n_items
        mean_user_size = np.mean(user_sizes)
        mean_item_size = np.mean(item_sizes)
        ratio = (mean_user_size * self.n_users + mean_item_size * self.n_items) / (num_entities * config.MAX_EMB_SIZE)
        txt = '{}; user weight: {:.2f}; alpha: ({:.2f}, {:.2f}); comp ratio: {:.4f}'
        return txt.format(self.distribution, self.user_weight, self.alphas[0], self.alphas[1], ratio)

