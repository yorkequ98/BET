from module import config
from util.train_util import logtxt
from module.Engine import Engine


IDs = {
    'gowalla': {
        # task ID and compression ratio
        0: 0.2
    }
}
config.BASE_MODEL = 'ngcf'
config.NUM_SAMPLES = 100


# use this to construction a population of actions
def contrust_population():
    for dataset_name in IDs:
        config.DATASET_NAME = dataset_name
        budgets = IDs[dataset_name]
        config.MAX_PATIENCE = {'yelp': 3, 'gowalla': 2}[dataset_name]

        for ID in budgets:

            logtxt('=' * 90)
            config.ID = ID

            config.BUDGET = budgets[ID]

            engine = Engine()

            logtxt('ID = {}; BUDGET = {}'.format(config.ID, config.BUDGET))
            engine.evo_search()


contrust_population()