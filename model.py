from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config



class SentimentLSTM(nn.Module):
    def __init__(self, config:Config, pad_token_id, device):
        super(SentimentLSTM, self).__init__()
        self.pad_token_id = pad_token_id
        self.device = device
        self.is_attn = config.is_attn
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=True)
        if self.is_attn:
            self.attention = Attention(self.hidden_size*2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()



    def init_hidden(self):
        h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size).to(self.device)
        return h0, c0


    def forward(self, x):
        self.batch_size = x.size(0)
        attn_output = None
        h0, c0 = self.init_hidden()

        x = self.embedding(x)
        x, _ = self.lstm(x, (h0, c0))
        if self.is_attn:
            attn_output = self.attention(self.relu(x))
            x = x * attn_output.unsqueeze(-1)
        x = torch.sum(x, dim=1)
        x = self.fc(x)

        return x.squeeze(1), attn_output



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/2), 1)
        )


    def forward(self, x):
        x = self.attention(x)
        x = x.squeeze(2)
        x = F.softmax(x, dim=1)
        return x



