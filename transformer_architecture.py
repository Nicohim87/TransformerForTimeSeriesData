import torch
import torch.nn as nn
import numpy as np
import copy
import math

class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim, mask=False):
        super(DotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.mask = mask

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, query, key, value):
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        score = torch.bmm(query, key.transpose(1, 2)) # (batch, d_seq, h_dim) * (batch, h_dim, e_seq) = (batch, d_seq, e_seq)
        score /= math.sqrt(query.shape[-1])

        # Masking
        if self.mask:
            mask = torch.triu(torch.ones(score.shape[-2:], device=score.device), diagonal=1)
            score = score.masked_fill(mask == 1, float('-inf'))

        # score -> (batch, d_seq, e_seq)
        score = torch.softmax(score, dim=-1) # Softmax along e_seq


        context_vector = torch.bmm(score, value) # (batch, d_seq, e_seq) * (batch, e_seq, h_dim) = (batch, d_seq, h_dim)
        return context_vector # (batch, d_seq, h_dim)
  

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim, mask=False):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.mask = mask

        self.attention = nn.ModuleList(
            [DotProductAttention(self.head_dim, mask) for _ in range(n_heads)]
        )

        self.fc_cat = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):

        head_outputs = []

        for i, attention_head in enumerate(self.attention):

            q = query[:, :, i*self.head_dim:(i+1)*self.head_dim]
            k = key[:, :, i*self.head_dim:(i+1)*self.head_dim]
            v = value[:, :, i*self.head_dim:(i+1)*self.head_dim]

            head_outputs.append(attention_head(q, k, v))

        context_vector = torch.cat(head_outputs, dim=2)

        return self.fc_cat(context_vector)
    

class PositionalFF(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super(PositionalFF, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, values):
        values = self.activation(self.fc1(values))
        return self.fc2(values)
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super(PositionalEncoding, self).__init__()
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


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, hidden_dim):
        super(EncoderBlock, self).__init__()
        
        self.multihead_attention = MultiheadAttention(n_heads, hidden_dim)
        self.feed_forward = PositionalFF(hidden_dim, int(hidden_dim*4))
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        attention = self.multihead_attention(x, x, x)
        x = self.norm1(x + attention)

        ff = self.feed_forward(x)
        x = self.norm2(x + ff)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderBlock, self).__init__()

        self.input_attention = MultiheadAttention(4, hidden_dim, mask=True)
        self.context_attention = MultiheadAttention(4, hidden_dim)
        self.feed_forward = PositionalFF(hidden_dim, int(hidden_dim*4))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, values, context_vector=None):
        input_attention = self.input_attention(values, values, values)
        values = self.norm1(values + input_attention)

        if context_vector:
            context_attention = self.context_attention(values, context_vector, context_vector)
        else:
            context_attention = self.context_attention(values, values, values)
        values = self.norm2(values + context_attention)

        ff = self.feed_forward(values)
        values = self.norm3(values + ff)

class Encoder(nn.Module):
    def __init__(self, corpus_size, hidden_dim, seq_len, n_blocks, n_heads, use_embedding=True, embedding_replacement:nn.Module=None):
        super(Encoder, self).__init__()
        if use_embedding:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(n_heads, hidden_dim) for _ in range(n_blocks)]
        )

    def forward(self, values):
        values = self.embedding(values)
        values = self.positional_encoding(values)

        for block in self.blocks:
            values = block(values)

        return values
    

class Decoder(nn.Module):
    def __init__(self, corpus_size, hidden_dim, seq_len, n_blocks, n_heads, use_embedding=True, embedding_replacement:nn.Module=None):
        super(Decoder, self).__init__()
        if use_embedding:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_heads, hidden_dim) for _ in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_dim, corpus_size)

    def forward(self, values, context_vector=None):
        values = self.embedding(values)
        values = self.positional_encoding(values)

        for block in self.blocks:
            values = block(values, context_vector)

        return self.fc_out(values)


class Transformer(nn.Module):
    def __init__(self, encoder_corpus_size, decoder_corpus_size, hidden_dim, seq_len):
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_corpus_size, hidden_dim, seq_len)
        self.decoder = Decoder(decoder_corpus_size, hidden_dim, seq_len)

    def encode(self, values):
        return self.encoder(values)

    def decode(self, values, context_vector):
        return self.decoder(values, context_vector)

    def forward(self, encoder_values, decoder_values):
        context_vector = self.encoder(encoder_values)
        return self.decoder(decoder_values, context_vector)