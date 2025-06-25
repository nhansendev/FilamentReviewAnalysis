import os
import random
import numpy as np

from torch import manual_seed
from torch.cuda import manual_seed_all

RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)  # legacy, unclear if useful
manual_seed(RANDOM_SEED)
manual_seed_all(RANDOM_SEED)

RNG = np.random.default_rng(RANDOM_SEED)

DATA_DIR = os.path.join(os.getcwd(), "Data")
