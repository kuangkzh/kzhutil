import torch
import numpy as np
import random


def set_seed(seed):
    """
    set seed for torch, numpy, random
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
