import torch
import torch.nn as nn
import numpy as np
import copy
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
    def __init__(self, n_heads, hidden_dim, dropout):
        super(EncoderBlock, self).__init__()
        
        self.multihead_attention = MultiheadAttention(n_heads, hidden_dim, dropout)
        self.feed_forward = PositionalFF(hidden_dim, int(hidden_dim*4), dropout)
        
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
    
    def forward(self, x):
        values = self.norm_attn(x)
        attention = self.multihead_attention(values, values, values)
        x = x + self.dropout_attn(attention)

        ff = self.feed_forward(self.norm_ff(x))
        x = x + self.dropout_ff(ff)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout):
        super(DecoderBlock, self).__init__()

        self.input_attention = MultiheadAttention(n_heads, hidden_dim, dropout, mask=True)
        self.context_attention = MultiheadAttention(n_heads, hidden_dim, dropout)
        self.feed_forward = PositionalFF(hidden_dim, int(hidden_dim*4), dropout)

        self.norm_selfattn = nn.LayerNorm(hidden_dim)
        self.norm_ctxattn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.dropout_selfattn = nn.Dropout(dropout)
        self.dropout_ctxattn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
    
    def forward(self, values, context_vector=None):
        norm_values = self.norm_selfattn(values)
        input_attention = self.input_attention(norm_values, norm_values, norm_values)
        values = values + self.dropout_selfattn(input_attention)

        norm_values = self.norm_ctxattn(values)
        if context_vector is not None:
            norm_context = self.norm_ctxattn(context_vector)
            context_attention = self.context_attention(norm_values, norm_context, norm_context)
        else:
            context_attention = self.context_attention(norm_values, norm_values, norm_values)
        values = values + self.dropout_ctxattn(context_attention)

        ff = self.feed_forward(self.norm_ff(values))
        values = values + self.dropout_ff(ff)
        return values

class Encoder(nn.Module):
    def __init__(self, corpus_size, hidden_dim, seq_len, n_blocks, n_heads, dropout=0.1, use_embedding=True, embedding_replacement:nn.Module=None):
        super(Encoder, self).__init__()
        if use_embedding:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(n_heads, hidden_dim, dropout) for _ in range(n_blocks)]
        )

    def forward(self, values):
        values = self.embedding(values)
        values = self.positional_encoding(values)

        for block in self.blocks:
            values = block(values)

        return values
    

class Decoder(nn.Module):
    def __init__(self, corpus_size, hidden_dim, seq_len, n_blocks, n_heads, dropout=0.1, use_embedding=True, embedding_replacement:nn.Module=None):
        super(Decoder, self).__init__()
        if use_embedding:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_heads, hidden_dim, dropout) for _ in range(n_blocks)]
        )

        self.head = nn.Linear(hidden_dim, corpus_size)

    def forward(self, values, context_vector=None):
        values = self.embedding(values)
        values = self.positional_encoding(values)

        for block in self.blocks:
            values = block(values, context_vector)

        return self.head(values)


class Transformer(nn.Module):
    def __init__(self, hidden_dim, seq_len, corpus_size, encoder_blocks, encoder_attention_heads, decoder_blocks, decoder_attention_heads, dropout=0.1, use_embedding=True, embedding_replacement:nn.Module=None):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus_size, hidden_dim, seq_len, encoder_blocks, encoder_attention_heads, dropout, use_embedding, embedding_replacement)
        self.decoder = Decoder(corpus_size, hidden_dim, seq_len, decoder_blocks, decoder_attention_heads, dropout, use_embedding, embedding_replacement)

    def encode(self, values):
        return self.encoder(values)

    def decode(self, values, context_vector):
        return self.decoder(values, context_vector)

    def forward(self, encoder_values, decoder_values):
        context_vector = self.encoder(encoder_values)
        return self.decoder(decoder_values, context_vector)