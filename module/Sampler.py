
from module import config
import numpy as np
import random
from scipy.stats import powerlaw, truncexpon, truncnorm, lognorm
from module.Action import Action


class Sampler:
    def __init__(self, dataset, num_actions):
        self.dataset = dataset
        self.num_actions = num_actions

    def random_ints_sum_to_m(self, alpha, dim, budget, distribution):
        """
        Randomly generating pvals ofr
        :param dim: number of entities
        :param budget: memory budget
        :param distribution: choice of distribution
        :return: embedding sizes
        """
        if distribution == 'powerlaw1' or distribution == 'powerlaw2':
            pvals = powerlaw.rvs(a=alpha, size=dim)
        elif distribution == 'exponential':
            pvals = truncexpon.rvs(b=alpha, size=dim)
        elif distribution == 'lognormal':
            pvals = lognorm.rvs(s=alpha, size=dim)
        else:
            assert distribution == 'normal'
            pvals = truncnorm.rvs(a=0, b=alpha, size=dim)
        # normalising
        pvals = pvals / sum(pvals)
        emb_sizes = np.floor(budget * pvals).astype(int)
        emb_sizes = np.maximum(config.MIN_EMB_SIZE, emb_sizes)
        return emb_sizes, pvals

    def choose_dist(self):
        max_alphas = {'powerlaw1': 20, 'lognormal': 0.5, 'normal': 20, 'exponential': 5}
        candidate_dists = list(config.DISTRIBUTIONS.keys())
        ps = list(config.DISTRIBUTIONS.values())
        choice = np.random.choice(candidate_dists, size=1, p=ps)[0]
        return choice, max_alphas[choice]

    def sample_one_action(self, budget, sort_size):
        stop = False
        user_ints = []
        item_ints = []
        user_alpha = -1
        item_alpha = -1
        user_pvals = []
        item_pvals = []
        user_dist = ''
        item_dist = ''
        n_users = self.dataset.n_users
        n_items = self.dataset.n_items

        user_weight = np.random.uniform(0, 1, size=1)[0]
        item_weight = 1 - user_weight
        while not stop:
            user_dist, max_user_alpha = self.choose_dist()
            item_dist, max_item_alpha = self.choose_dist()
            user_alpha = np.random.uniform(0, max_user_alpha)
            item_alpha = np.random.uniform(0, max_item_alpha)

            user_ints, user_pvals = self.random_ints_sum_to_m(user_alpha, n_users, user_weight * budget, user_dist)
            item_ints, item_pvals = self.random_ints_sum_to_m(item_alpha, n_items, item_weight * budget, item_dist)

            if max(user_ints) <= config.MAX_EMB_SIZE and max(item_ints) <= config.MAX_EMB_SIZE:
                stop = True
        user_ints.sort()
        item_ints.sort()
        user_ints = user_ints[::-1]
        item_ints = item_ints[::-1]

        assert len(user_pvals) > 0 and len(item_pvals) > 0
        return Action(
            self.dataset,
            (user_alpha, item_alpha),
            (user_ints, item_ints),
            (user_dist, item_dist),
            user_weight,
            (user_pvals, item_pvals),
            sort_size
        )

    def action_sampler(self, budget):
        random_actions = []
        for _ in range(self.num_actions):
            action = self.sample_one_action(budget, sort_size=True)
            random_actions.append(action)
        random.shuffle(random_actions)
        return random_actions


