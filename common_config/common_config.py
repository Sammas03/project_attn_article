

import torch

parents_config = {
    'gpu': 1 if torch.cuda.is_available() else 0
}