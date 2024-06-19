import torch
import torch.nn as nn
import torch.nn.functional as F



class SentimentLSTM(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(SentimentLSTM, self).__init__()
        self.device = device
        self.use_attention = config.use_attention
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_layer = config.num_layer
        self.dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layer,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=True)
        if self.use_attention:
            self.attention = Attention(self.hidden_dim*2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim*2, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()



    def init_hidden(self):
        h0 = torch.zeros(self.num_layer*2, self.batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layer*2, self.batch_size, self.hidden_dim).to(self.device)
        return h0, c0


    def forward(self, x):
        self.batch_size = x.size(0)
        attn_output = None
        h0, c0 = self.init_hidden()

        x = self.embedding(x)
        x, _ = self.lstm(x, (h0, c0))
        if self.use_attention:
            attn_output = self.attention(self.relu(x))
            x = x * attn_output.unsqueeze(-1)
        x = torch.sum(x, dim=1)
        x = self.fc(x)

        return x.squeeze(1), attn_output



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, int(self.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim/2), 1)
        )


    def forward(self, x):
        x = self.attention(x)
        x = x.squeeze(2)
        x = F.softmax(x, dim=1)
        return x



