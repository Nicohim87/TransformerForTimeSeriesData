from transformer_components import *
import torch.nn as nn

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
        if use_embedding or not embedding_replacement:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = ParameterPositionalEncoding(hidden_dim, seq_len)
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
        self.positional_encoding = ParameterPositionalEncoding(hidden_dim, seq_len)
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
    def __init__(self, hidden_dim, seq_len, corpus_size, 
                 encoder_blocks, encoder_attention_heads, 
                 decoder_blocks, decoder_attention_heads, 
                 dropout=0.1, 
                 use_embedding=True, embedding_replacement:nn.Module=None):
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