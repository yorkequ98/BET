import torch
import os
import sys

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
MAX_EMB_SIZE = 128
MIN_EMB_SIZE = 1
SEED = 42
BATCH_SIZE = 10000
MAX_PATIENCE = 3
BASE_MODEL = ''
DECAY_BATCHES = 200
MIN_LR = 0.001
MAX_LR = 0.03
BUDGET = ''

DATASET_NAME = None
CHUNK_NUM = 1
TRAINING_EPOCHS = 100000

NUM_SAMPLES = -1
VERBOSITY = 0
FINE_TUNE_EPOCHS = 10
EVO_SEARCH_ITERATIONS = 50
MAX_RQ = {'yelp': 0, 'ml-1m': 0, 'ml-25m': 0, 'gowalla': 0}
DISTRIBUTIONS = {'powerlaw1': 1/4, 'normal': 1/4, 'exponential': 1/4, 'lognormal': 1/4}
ID = -1
MIX_DIST = True
USE_DUMMY = False

# Default column names
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_LABEL_COL = "label"
DEFAULT_TITLE_COL = "title"
DEFAULT_GENRE_COL = "genre"
DEFAULT_RELEVANCE_COL = "relevance"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_SIMILARITY_COL = "sim"
DEFAULT_ITEM_FEATURES_COL = "features"
DEFAULT_ITEM_SIM_MEASURE = "item_cooccurrence_count"

DEFAULT_HEADER = (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

COL_DICT = {
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_prediction": DEFAULT_PREDICTION_COL,
}