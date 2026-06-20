from transformer_components import *
from transformer_architecture_base import EncoderBlock, DecoderBlock
import torch.nn as nn
import copy

class SwitchLayer(nn.Module):
    def __init__(self, hidden_dim, dropout, expert_network:nn.Module, n_expert:int=4, top_k:int=1):
        super().__init__()

        self.router = nn.Linear(hidden_dim, n_expert)
        self.noise = nn.Linear(hidden_dim, n_expert)
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.xavier_uniform_(self.noise.weight)
        nn.init.zeros_(self.noise.bias)

        self.top_k = top_k

        self.experts = nn.ModuleList(
            [copy.deepcopy(expert_network).to("cuda") for _ in range(n_expert)]
        )
        self.expert_count = [0]*n_expert

        self.softplus = nn.Softplus()
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    
    def keep_top_k(self, expert_logits):
        values, indices = torch.topk(expert_logits, self.top_k, dim=-1)

        logits = torch.full_like(expert_logits, float('-inf'))
        logits.scatter_(1, indices, values)

        return logits
    
    def get_expert_count(self):
        return self.expert_count.copy()
    
    def reset_expert_count(self):
        self.expert_count = [0 for _ in self.expert_count]

    def forward(self, x):

        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        x_float = x_flat.float()

        expert_logits = self.router(x_float)
        noise = torch.randn_like(expert_logits)*self.softplus(self.noise(x_float))
        expert_logits = self.activation(self.keep_top_k(expert_logits + noise))
        expert_logits = expert_logits.to(x_flat.dtype)

        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):

            idx = (expert_logits[:, i] > 0).nonzero(as_tuple=True)[0]

            if idx.numel() == 0:
                continue

            tokens = x_flat[idx]
            weights = expert_logits[idx, i].unsqueeze(-1)

            expert_out = expert(tokens)

            output[idx] += weights * expert_out

            # Track expert usage (no grad)
            self.expert_count[i] += idx.numel()

        return self.dropout(output.reshape(B, S, D))

class MoE_EncoderBlock(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout, expert_network, n_expert, top_k):
        super(MoE_EncoderBlock, self).__init__()
        
        self.multihead_attention = MultiheadAttention(n_heads, hidden_dim, dropout)
        self.feed_forward = SwitchLayer(hidden_dim, dropout, expert_network, n_expert, top_k)
        
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
    
class MoE_DecoderBlock(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout, expert_network, n_expert, top_k):
        super(MoE_DecoderBlock, self).__init__()

        self.input_attention = MultiheadAttention(n_heads, hidden_dim, dropout, mask=True)
        self.context_attention = MultiheadAttention(n_heads, hidden_dim, dropout)
        self.feed_forward = SwitchLayer(hidden_dim, dropout, expert_network, n_expert, top_k)

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

class MoE_Encoder(nn.Module):
    def __init__(self, corpus_size, hidden_dim, seq_len, n_blocks, n_heads, expert_network, n_expert, top_k, dropout=0.1, use_embedding=True, embedding_replacement:nn.Module=None):
        super(MoE_Encoder, self).__init__()
        if use_embedding or not embedding_replacement:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = ParameterPositionalEncoding(hidden_dim, seq_len)
        self.blocks = nn.ModuleList(
            [MoE_EncoderBlock(n_heads, hidden_dim, dropout, expert_network, n_expert, top_k) for i in range(n_blocks)]
        )

    def forward(self, values):
        values = self.embedding(values)
        values = self.positional_encoding(values)

        for block in self.blocks:
            values = block(values)

        return values
    

class MoE_Decoder(nn.Module):
    def __init__(self, corpus_size, hidden_dim, seq_len, n_blocks, n_heads, expert_network, n_expert, top_k, dropout=0.1, use_embedding=True, embedding_replacement:nn.Module=None):
        super(MoE_Decoder, self).__init__()
        if use_embedding:
            self.embedding = nn.Embedding(corpus_size, hidden_dim)
        else:
            self.embedding = embedding_replacement
        self.positional_encoding = ParameterPositionalEncoding(hidden_dim, seq_len)
        self.blocks = nn.ModuleList(
            [MoE_DecoderBlock(n_heads, hidden_dim, dropout, expert_network, n_expert, top_k) for _ in range(n_blocks)]
        )

        self.head = nn.Linear(hidden_dim, corpus_size)

    def forward(self, values, context_vector=None):
        values = self.embedding(values)
        values = self.positional_encoding(values)

        for block in self.blocks:
            values = block(values, context_vector)

        return self.head(values)


class MoE_Transformer(nn.Module):
    def __init__(self, hidden_dim, seq_len, corpus_size, 
                 encoder_blocks, encoder_attention_heads, 
                 decoder_blocks, decoder_attention_heads, 
                 encoder_expert_network:nn.Module=None, encoder_n_expert:int=4, encoder_top_k:int=1, 
                 decoder_expert_network:nn.Module=None, decoder_n_expert:int=4, decoder_top_k:int=1, 
                 dropout=0.1, 
                 use_embedding=True, embedding_replacement:nn.Module=None):
        super(MoE_Transformer, self).__init__()

        if not encoder_expert_network:
            encoder_expert_network = PositionalFF(hidden_dim, int(4*hidden_dim), dropout)

        if not decoder_expert_network:
            decoder_expert_network = PositionalFF(hidden_dim, int(4*hidden_dim), dropout)

        self.encoder = MoE_Encoder(corpus_size, hidden_dim, seq_len, encoder_blocks, encoder_attention_heads, encoder_expert_network, encoder_n_expert, encoder_top_k, dropout, use_embedding, embedding_replacement)
        self.decoder = MoE_Decoder(corpus_size, hidden_dim, seq_len, decoder_blocks, decoder_attention_heads, decoder_expert_network, decoder_n_expert, decoder_top_k, dropout, use_embedding, embedding_replacement)

    def encode(self, values):
        return self.encoder(values)

    def decode(self, values, context_vector):
        return self.decoder(values, context_vector)

    def forward(self, encoder_values, decoder_values):
        context_vector = self.encoder(encoder_values)
        return self.decoder(decoder_values, context_vector)