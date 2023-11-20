import os
import torch
import random
import numpy as np


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  
    np.random.seed(seed)
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
