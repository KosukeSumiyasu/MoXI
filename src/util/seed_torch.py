import os
import random
import numpy as np
import torch

# set seed
def seed_torch(seed) -> None:
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'backends'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False