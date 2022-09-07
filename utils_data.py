import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
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