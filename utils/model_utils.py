import numpy as np
import torch
from neuralprophet import set_random_seed


def set_seeds(random_seed=4):
    set_random_seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.set_num_threads(1)  # Limit Torch to use only one thread
