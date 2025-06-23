from module import config
import os
import pickle as pkl
from module.Population import Population
import numpy as np
from module.Action import Action


def clear_output_log():
    folder = 'tmp/logs/{}/'.format(config.ID)
    exist = os.path.exists(folder)
    if not exist:
        os.makedirs(folder)
    with open(folder + 'output.txt', 'w') as f:
        f.write('\n' * 10)


def clear_dist_and_losses():
    ID = config.ID
    folder = 'tmp/dist/{}/'.format(ID)
    if not os.path.exists(folder):
        os.makedirs(folder)
    folder = 'tmp/losses/{}/'.format(ID)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open('tmp/dist/{}/dist.txt'.format(ID), 'a') as f:
        f.write('\n' * 10)
    with open('tmp/losses/{}/losses.txt'.format(ID), 'a') as f:
        f.write('\n' * 10)


def logtxt(txt):
    print(txt)
    if config.VERBOSITY != 0:
        folder = 'tmp/logs/{}/'.format(config.ID)
        exist = os.path.exists(folder)
        if not exist:
            os.makedirs(folder)

        with open(folder + 'output.txt', 'a', encoding="utf-8") as f:
            txt += '\n'
            f.write(txt)


def save_progress(step, population, selected_action, losses=None):
    # save population
    folder = 'tmp/population/{}/'.format(config.ID)
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamps = ''
    for action in population.actions:
        action = action[0]
        timestamp = action.timestamp
        timestamps += str(timestamp)
        timestamps += '\n'
    with open(folder + 'step_{}.txt'.format(step), 'a') as file:
        file.write(timestamps)
    # save selected action
    folder = 'tmp/action_txt/{}/'.format(config.ID)
    if not os.path.exists(folder):
        os.makedirs(folder)
    serialize_action(selected_action, config.ID)

    distributions = selected_action.distribution
    with open('tmp/dist/{}/dist.txt'.format(config.ID), 'a') as f:
        f.write(str(distributions) + '\n')

    if losses is not None:
        losses = [str(loss) for loss in losses]
        with open('tmp/losses/{}/losses.txt'.format(config.ID), 'a') as f:
            to_write = ', '.join(losses)
            to_write += '\n'
            f.write(to_write)


def load_progress(step, dataset):
    try:
        path = 'tmp/population/{}/step_{}.pkl'.format(config.ID, step)
        with open(path, 'rb') as file:
            population = pkl.load(file)
            return population
    except:
        path = 'tmp/population/{}/step_{}.txt'.format(config.ID, step)
        popu_path = path.format(step)
        population = Population()
        with open(popu_path, 'r') as file:
            # population = pkl.load(file)
            timestamps = file.readlines()
        for timestamp in timestamps:
            timestamp = timestamp.split('\n')[0]
            path = 'tmp/action_txt/{}/{}.txt'.format(config.ID, timestamp)
            action = load_action(path, dataset, timestamp)
            population.actions.append((action, None))
        return population


def serialize_action(action, ID):
    # makedir if not exist
    if not os.path.exists('tmp/action_txt/{}/'.format(ID)):
        os.makedirs('tmp/action_txt/{}/'.format(ID))

    # clear existing files
    open('tmp/action_txt/{}/{}.txt'.format(ID, action.timestamp), 'w').close()

    with open('tmp/action_txt/{}/{}.txt'.format(ID, action.timestamp), 'a') as file:
        # user sizes
        for size in action.ints[0]:
            file.write(str(size) + ' ')
        file.write('|')
        # item sizes
        for size in action.ints[1]:
            file.write(str(size) + ' ')
        file.write('\n')

        # distributions
        file.write(action.distribution[0] + '|' + action.distribution[1])
        file.write('\n')

        # user weight
        file.write(str(action.user_weight))
        file.write('\n')

        # user pvals
        for pval in action.pvals[0]:
            file.write(str(pval) + ' ')
        file.write('|')

        # item pvals
        for pval in action.pvals[1]:
            file.write(str(pval) + ' ')
        file.write('\n')

        # alpha
        file.write(str(action.alphas[0]) + '|' + str(action.alphas[1]))
        file.write('\n')

        # metric
        file.write(str(action.metric))
        file.write('\n')

        # sort sizes
        file.write(str(action.sort_sizes))


def load_action(path, dataset, timestamp):
    with open(path, 'r') as file:
        lines = file.readlines()

    ints_line = lines[0]
    user_ints_txt, item_ints_txt = ints_line.split('|')
    user_sizes = user_ints_txt.strip().split(' ')
    item_sizes = item_ints_txt.strip().split(' ')
    user_sizes = np.array([int(size) for size in user_sizes])
    item_sizes = np.array([int(size) for size in item_sizes])
    ints = (user_sizes, item_sizes)

    dist_line = lines[1]
    user_dist, item_dist = dist_line.strip('\n').split('|')
    distribution = (user_dist, item_dist)

    user_weight = float(lines[2])

    pvals_line = lines[3]
    user_pval_txt, item_pval_txt = pvals_line.split('|')
    user_pvals = user_pval_txt.strip().split(' ')
    item_pvals = item_pval_txt.strip().split(' ')
    user_pvals = np.array([float(pval) for pval in user_pvals])
    item_pvals = np.array([float(pval) for pval in item_pvals])
    pvals = (user_pvals, item_pvals)

    user_alpha, item_alpha = lines[4].split('|')
    user_alpha = float(user_alpha)
    item_alpha = float(item_alpha)
    alpha = (user_alpha, item_alpha)

    metric = float(lines[5])

    sort_sizes = bool(lines[6])

    action = Action(dataset, alpha, ints, distribution, user_weight, pvals, sort_sizes)

    action.timestamp = int(timestamp)
    action.metric = metric
    return action
