from module import config
import time
from util.train_util import initialize_dataset, train_till_convergence
from util.eval_util import eval_rec
from util.IO_util import load_progress, logtxt, clear_output_log
import os
from base_recsys.LightGCN import LightGCN
from base_recsys.NGCF import NGCF
from base_recsys.NCF import NeuMF
import numpy as np
from queue import PriorityQueue
from module.Action import Action


class PopulationListener:
    def __init__(self, evaluated=None):
        if evaluated is not None:
            self.evaluated = evaluated
        else:
            self.evaluated = []
        self.best_actions = PriorityQueue()

        self.dataset = initialize_dataset()

    def eval_rec_with_action(self, action, dataset):
        user_sizes, item_sizes = action.get_array_form()
        # initialise a new recommender
        if config.BASE_MODEL == 'lightgcn':
            recsys = LightGCN(dataset, user_sizes, item_sizes).to(config.device)
        elif config.BASE_MODEL == 'ngcf':
            recsys = NGCF(dataset, user_sizes, item_sizes).to(config.device)
        elif config.BASE_MODEL == 'neumf':
            recsys = NeuMF(dataset, user_sizes, item_sizes).to(config.device)
        else:
            pass

        folder = 'tmp/model/{}/'.format(config.ID)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = folder + 'actiontimestep_{}'.format(action.timestamp)

        _ = train_till_convergence(recsys, dataset, path)

        rq = eval_rec(recsys, dataset, 'test')
        return rq

    def listen_to_population(self, steps, k=3):
        config.VERBOSITY = 1
        clear_output_log()

        previous_e = ''
        while len(steps) > 0:
            step = steps[0]
            try:
                population = load_progress(step, self.dataset)
                steps.pop(0)
            except Exception as e:
                if previous_e != str(e):
                    previous_e = str(e)
                    print(e)
                time.sleep(2.0)
                continue

            # print information of the top k actions
            top_actions, _ = population.get_best_action(k)
            top_timestamps = [action.timestamp for action in top_actions]
            logtxt('*' * 35 + 'step {}: {}'.format(step, str(top_timestamps)) + '*' * 35)
            # print the top-3 actions
            for action in top_actions:
                txt = 'timestamp {}; {}; RQ： {:.4f}'.format(
                    action.timestamp, str(action), action.metric
                )
                user_sizes, item_sizes = action.ints

                print(user_sizes, np.sum(user_sizes))
                capped_user_sizes = np.minimum(np.ones_like(user_sizes) * 128, user_sizes)
                extra_memo = np.sum(user_sizes - capped_user_sizes)
                avg = extra_memo // np.sum(user_sizes < 64)
                print(avg)
                capped_user_sizes[-np.sum(user_sizes < 64):] += avg
                print(capped_user_sizes, np.sum(capped_user_sizes), np.max(capped_user_sizes))

                print(item_sizes, np.sum(item_sizes))
                capped_item_sizes = np.minimum(np.ones_like(item_sizes) * 64, item_sizes)
                extra_memo = np.sum(item_sizes - capped_item_sizes)
                avg = extra_memo // np.sum(item_sizes < 32)
                print(avg)
                capped_item_sizes[-np.sum(item_sizes < 32):] += avg
                print(capped_item_sizes, np.sum(capped_item_sizes), np.max(capped_item_sizes))

                # topindex = int(len(user_sizes) * 0.1)
                # tmp = user_sizes[:topindex]
                # user_sizes[:topindex] = user_sizes[4 * topindex: 5 * topindex]
                # user_sizes[4 * topindex: 5 * topindex] = tmp

                # topindex = int(len(item_sizes) * 0.1)
                # tmp = item_sizes[:topindex]
                # item_sizes[:topindex] = item_sizes[4 * topindex: 5 * topindex]
                # item_sizes[4 * topindex: 5 * topindex] = tmp

                memory_size = np.mean(user_sizes) * self.dataset.n_users + np.mean(item_sizes) * self.dataset.n_items
                max_memory = config.MAX_EMB_SIZE * (self.dataset.n_users + self.dataset.n_items)
                logtxt('compression ratio: {:.4f}'.format(memory_size / max_memory))
                logtxt(txt)

            for action in top_actions:
                if action.timestamp not in self.evaluated:
                    txt = 'timestamp {}'.format(action.timestamp)
                    logtxt('-' * 30 + txt + '-' * 30)
                    self.evaluated.append(action.timestamp)

                    rq = self.eval_rec_with_action(action, self.dataset)

                    self.best_actions.put((rq, action))

        print('num of actions evaluated: ', self.best_actions.qsize(), len(self.evaluated))
        logtxt('========== {} =========='.format('THE END'))

    def evaluate_action(self, action, ratio):
        budget = ratio * (self.dataset.n_items + self.dataset.n_users) * config.MAX_EMB_SIZE
        user_pvals, item_pvals = action.pvals
        user_ints = np.maximum(1, np.round(budget * user_pvals)).astype(int)
        item_ints = np.maximum(1, np.round(budget * item_pvals)).astype(int)
        user_ints.sort()
        item_ints.sort()
        user_ints = user_ints[::-1]
        item_ints = item_ints[::-1]

        action = Action(
            self.dataset.n_users, self.dataset.n_items,
            action.alphas,
            (user_ints, item_ints),
            action.distribution,
            action.user_weight,
            action.pvals
        )
        self.eval_rec_with_action(action, self.dataset)
