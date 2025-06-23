from module import config
from module.DeepSets import DeepSets
import numpy as np
import torch
from module.Population import Population
from util.eval_util import eval_rec
from util.IO_util import logtxt, save_progress, clear_dist_and_losses
from util.train_util import initialize_dataset, train_T_epochs, get_pretrained_recsys
from module.ReplyBuffer import ReplyBuffer
from module.Sampler import Sampler


class Engine:
    def __init__(self):
        self.dataset = initialize_dataset(with_validation=True)
        self.population = Population()
        user_freq = list(self.dataset.user_freq.values())
        item_freq = list(self.dataset.item_freq.values())
        self.critic = DeepSets(user_freq, item_freq).to(config.device)
        self.buffer = ReplyBuffer()

    def evo_search(self):
        config.VERBOSITY = 0
        clear_dist_and_losses()

        budget = config.BUDGET * config.MAX_EMB_SIZE * (self.dataset.n_users + self.dataset.n_items)

        sampler = Sampler(self.dataset, config.NUM_SAMPLES)

        self.critic.train()
        for step in range(config.EVO_SEARCH_ITERATIONS):
            logtxt('-' * 35 + 'step {}'.format(step) + '-' * 35)

            sampled_actions = sampler.action_sampler(budget)

            action_selected, emb_selected, pred_score = self.select_action(sampled_actions, step)

            rq = self.probe_action(action_selected)

            # normalise RQ
            normalised_rq = rq / config.MAX_RQ[config.DATASET_NAME]

            self.buffer.add(action_selected, normalised_rq)
            self.population.add_action(action_selected, emb_selected, normalised_rq)

            losses = self.critic.update_critic(self.buffer)
            critic_loss = sum(losses) / 2

            message = 'Step {}, predicted RQ: {:.4f}, actual RQ: {:.4f}, critic loss: {:.4f}'
            logtxt(message.format(
                step, pred_score.item(), normalised_rq, critic_loss
            ))

            save_progress(step, action_selected, losses)

    def probe_action(self, action):
        # initialise a pretrained recommender
        recsys = get_pretrained_recsys(self.dataset)
        assert np.mean(recsys.user_sizes) == config.MAX_EMB_SIZE and np.mean(recsys.item_sizes) == config.MAX_EMB_SIZE
        # update the embedding sizes according to the action
        user_sizes, item_sizes = action.ints
        recsys.update_sizes(user_sizes, item_sizes)
        # fine-tuning
        recsys = train_T_epochs(recsys, self.dataset, config.FINE_TUNE_EPOCHS)

        sampling_ratio = {'gowalla': 0.5, 'yelp': 0.5, 'ml-1m': 1.0}[config.DATASET_NAME]
        rq = eval_rec(recsys, self.dataset, sampling_ratio)
        return rq

    def get_action_emb_and_score(self, action):
        assert isinstance(self.critic, DeepSets)
        users, items, mask, num_nonzeros = action.get_set_form()
        action_embed, score = self.critic.forward(users, items, mask, num_nonzeros)
        return action_embed, score

    def select_optimal_action(self, sampled_actions):
        current_best = -1e10
        optimal_action = None
        optimal_emb = None
        m = len(sampled_actions)
        for action in sampled_actions:
            action_embed, score = self.get_action_emb_and_score(action)
            if score >= current_best:
                current_best = score
                optimal_action = action
                optimal_emb = action_embed
        logtxt('Selected optimal action: {} from {} actions'.format(str(optimal_action), m))
        return optimal_action, optimal_emb, current_best

    def select_nearest_action(self, sampled_actions):
        _, centroid = self.population.get_best_action(1)
        centroid = centroid[0]
        nearest_action = None
        nearest_emb = None
        current_nearest = 1e10
        associated_score = None
        m = len(sampled_actions)
        for action in sampled_actions:
            action_embed, score = self.get_action_emb_and_score(action)
            euclidian = torch.nn.PairwiseDistance(p=2)
            sim = euclidian(centroid, action_embed).item()
            if sim <= current_nearest:
                current_nearest = sim
                nearest_action = action
                nearest_emb = action_embed
                associated_score = score
        logtxt('Selected nearest action: {} from {} actions'.format(str(nearest_action), m))
        return nearest_action, nearest_emb, associated_score

    def select_random_action(self, sampled_actions):
        action_rand = np.random.choice(sampled_actions, 1)[0]
        emb_rand, score_rand = self.get_action_emb_and_score(action_rand)
        m = len(sampled_actions)
        logtxt('Selected random action: {} from {} actions'.format(str(action_rand), m))
        return action_rand, emb_rand, score_rand

    def select_action(self, sampled_actions, step):
        selected_method = step % 5

        if selected_method in [0, 1, 2]:
            # predicted fitness selection
            action_selected, emb_selected, pred_score = self.select_optimal_action(sampled_actions)
        elif selected_method in [3]:
            # nearest neighbour selection
            action_selected, emb_selected, pred_score = self.select_nearest_action(sampled_actions)
        else:
            assert selected_method == 4
            # random selection
            action_selected, emb_selected, pred_score = self.select_random_action(sampled_actions)

        action_selected.timestamp = step
        return action_selected, emb_selected, pred_score

