import torch
import torch.nn as nn
import numpy as np
import math

class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim, dropout, mask=False):
        super(DotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.mask = mask
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value):

        score = torch.bmm(query, key.transpose(1, 2)) # (batch, d_seq, h_dim) * (batch, h_dim, e_seq) = (batch, d_seq, e_seq)
        score /= math.sqrt(query.shape[-1])

        # Masking
        if self.mask:
            mask = torch.triu(torch.ones(score.shape[-2:], device=score.device), diagonal=1)
            score = score.masked_fill(mask == 1, float('-inf'))

        # score -> (batch, d_seq, e_seq)
        score = torch.softmax(score, dim=-1) # Softmax along e_seq
        score = self.dropout(score)


        context_vector = torch.bmm(score, value) # (batch, d_seq, e_seq) * (batch, e_seq, h_dim) = (batch, d_seq, h_dim)
        return context_vector # (batch, d_seq, h_dim)
  

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout, mask=False):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.mask = mask
    
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.ModuleList(
            [DotProductAttention(self.head_dim, dropout, mask) for _ in range(n_heads)]
        )

        self.fc_cat = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):

        head_outputs = []

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        for i, attention_head in enumerate(self.attention):

            q = query[:, :, i*self.head_dim:(i+1)*self.head_dim]
            k = key[:, :, i*self.head_dim:(i+1)*self.head_dim]
            v = value[:, :, i*self.head_dim:(i+1)*self.head_dim]

            head_outputs.append(attention_head(q, k, v))

        context_vector = torch.cat(head_outputs, dim=2)

        return self.dropout(self.fc_cat(context_vector))
    

class PositionalFF(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout):
        super(PositionalFF, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, values):
        values = self.activation(self.fc1(values))
        values = self.dropout(values)
        return self.fc2(values)
    

class SinusodialPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super(SinusodialPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        encoding = np.zeros((1, max_len, hidden_dim))
        position = np.arange(max_len).reshape(-1, 1)
        div_term = 1 / (10000 ** (np.arange(0, hidden_dim, 2) / hidden_dim))

        encoding[:, :, 0::2] = np.sin(position * div_term)
        encoding[:, :, 1::2] = np.cos(position * div_term)

        self.encoding = torch.tensor(encoding, dtype=torch.float32).detach()

    def forward(self, values):
        return values + self.encoding[:, :values.shape[1], :].to(values.device)
    
class ParameterPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super(ParameterPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.encoding = nn.Parameter(torch.randn(1, self.max_len, hidden_dim))

    def forward(self, values):
        return values + self.encoding[:, :values.shape[1], :].to(values.device)