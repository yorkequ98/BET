import os
import math
from datasets.ImplicitCF import ImplicitCF
from module import config
from util.eval_util import eval_rec
from util.IO_util import logtxt
import torch
import numpy as np
from base_recsys.LightGCN import LightGCN
from base_recsys.NGCF import NGCF
from base_recsys.NCF import NeuMF
import torch.optim as optim


def initialize_dataset(with_validation):
    base_model = config.BASE_MODEL
    dataset_type = config.DATASET_NAME
    # setting up the dataset
    dataset = ImplicitCF(base_model, dataset_type, with_validation=with_validation)

    logtxt('training set size = {0}, test set size = {1}, batch size = {2}'.format(
        dataset.train_size,
        dataset.test_size,
        config.BATCH_SIZE)
    )
    logtxt('user size: {}, item size: {}'.format(
        dataset.n_users,
        dataset.n_items
    ))
    return dataset


def train_one_batch(recsys, optimizer, dataset):
    users, pos_items, neg_items = dataset.train_loader(config.BATCH_SIZE)
    optimizer.zero_grad()
    total_loss, mf_loss, emb_loss = recsys.create_bpr_loss(
        users, pos_items, neg_items
    )
    total_loss.backward()
    optimizer.step()
    return total_loss, mf_loss, emb_loss


def train_n_steps(recsys, rec_opt, dataset, step_num):
    total_loss = []
    for batch in range(step_num):
        batch_loss, mf_loss, emb_loss = train_one_batch(recsys, rec_opt, dataset)
        total_loss.append(batch_loss)
        if (batch + 1) % config.DECAY_BATCHES == 0:
            rec_opt.param_groups[0]['lr'] = max(config.MIN_LR, 0.95 * rec_opt.param_groups[0]['lr'])
    assert total_loss != 0
    return sum(total_loss) / step_num


def train_T_epochs(recsys, dataset, T):
    rec_opt = optim.Adam(recsys.parameters(), lr=config.MAX_LR)

    # number of batches in each epoch
    batch_num = math.ceil(dataset.train_size / config.BATCH_SIZE)

    epoch_num = 0
    # number of batches (steps) in T epochs
    total_steps = T * batch_num
    for step_num in range(total_steps):
        # train one epoch
        batch_loss, mf_loss, emb_loss = train_one_batch(recsys, rec_opt, dataset)

        if (step_num + 1) % config.DECAY_BATCHES == 0:
            rec_opt.param_groups[0]['lr'] = max(config.MIN_LR, 0.98 * rec_opt.param_groups[0]['lr'])

        # end of epoch
        if (step_num + 1) % batch_num == 0:
            epoch_num += 1
            if epoch_num == T:
                message = 'Epoch %d, step num %d: [total loss %.5f = mf loss %.5f + emb loss %.5f]'
                logtxt(message % (epoch_num, step_num, batch_loss, mf_loss, emb_loss))

    return recsys


def train_till_convergence(recsys, dataset, path):
    rec_opt = optim.Adam(recsys.parameters(), lr=config.MAX_LR)
    patience = config.MAX_PATIENCE
    best_metric = {'ml-25m': 1e10, 'yelp': 0, 'ml-1m': 0, 'gowalla': 0}[config.DATASET_NAME]
    sampling_ratio = {
        'ml-25m': [0.1, 0.5], 'yelp': [0.1, 1.0], 'ml-1m': [1.0, 1.0], 'gowalla': [0.1, 1.0]
    }[config.DATASET_NAME]

    recsys.train()
    for i in range(100000):
        loss = train_n_steps(recsys, rec_opt, dataset, 2000)
        logtxt('i: {}. loss: {:.4f}'.format(i, loss))
        metric = eval_rec(recsys, dataset, user_sample_ratio=sampling_ratio[0])
        if metric >= best_metric:
            best_metric = metric
            patience = config.MAX_PATIENCE
            torch.save(recsys.state_dict(), path + '.pth')
            torch.save(recsys.state_dict(), path + '_optimiser.pth')
        else:
            patience -= 1
            logtxt('Current patience： {}'.format(patience))
            if patience <= 0:
                break
    logtxt('Loading from {}'.format(path + '.pth'))
    recsys.load_state_dict(torch.load(path + '.pth'))

    return best_metric


def get_pretrained_recsys(dataset):
    user_sizes = np.ones(dataset.n_users, dtype=int) * config.MAX_EMB_SIZE
    item_sizes = np.ones(dataset.n_items, dtype=int) * config.MAX_EMB_SIZE
    recsys = get_fresh_recommender(dataset, user_sizes, item_sizes)
    folder = 'tmp/pretrained/{}/{}/'.format(config.DATASET_NAME, config.BASE_MODEL)
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = folder + 'trained_{}_random_False'.format(1.0)
    try:
        recsys.load_state_dict(torch.load(path + '.pth'))
    except Exception as e:
        print(e)
        print('Failed to load the pretrained model. Now pretraining the model')
        train_till_convergence(recsys, dataset, path)

    if config.MAX_RQ[config.DATASET_NAME] == 0:
        print('Computing MAX RQ...')
        config.MAX_RQ[config.DATASET_NAME] = eval_rec(recsys, dataset, 1.0)
    return recsys


def get_fresh_recommender(dataset, user_sizes, item_sizes):
    if config.BASE_MODEL == 'lightgcn':
        recsys = LightGCN(dataset, user_sizes, item_sizes).to(config.device)
    elif config.BASE_MODEL == 'ngcf':
        recsys = NGCF(dataset, user_sizes, item_sizes).to(config.device)
    elif config.BASE_MODEL == 'ncf':
        recsys = NeuMF(dataset, user_sizes, item_sizes).to(config.device)
    else:
        raise ValueError('Invalid choice of base model: {}'.format(config.BASE_MODEL))
    return recsys
