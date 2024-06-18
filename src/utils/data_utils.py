import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import LOGGER, colorstr



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, config, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        
        self.text, self.label = [], []
        for text, label in self.data:
            self.label.append(label)
            t = self.tokenizer.encode(text)[:self.max_len]
            t = t + [self.tokenizer.pad_token_id]*(self.max_len-len(t))
            self.text.append(t)

        assert len(self.text) == len(self.label)
        self.length = len(self.label)


    def __getitem__(self, idx):
        return torch.LongTensor(self.text[idx]), torch.tensor(self.label[idx], dtype=torch.float)

    
    def __len__(self):
        return self.length
    

class CustomDLoader(Dataset):
    def __init__(self, path):
        LOGGER.info(colorstr('red', 'Custom dataloader is required..'))
        raise NotImplementedError

    def __getitem__(self, idx):
        pass
    
    def __len__(self):
        pass