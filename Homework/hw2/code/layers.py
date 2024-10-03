import torch
import torch.nn as nn
import math
from utils import clones
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
def attention(query, key, value, mask=None, dropout=None):
    # Your code here
    # get dimension of keys/queries
    d_k = query.size(-1)
    
    # dot product btwn query & key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # apply mask if needed
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    # attention weights w/ softmax
    attn = torch.softmax(scores, dim=-1)
    
    # apply dropout if needed
    if dropout is not None:
        attn = dropout(attn)
        
    # attention weights to get weighted sum of vals
    out = torch.matmul(attn, value)
    
    return out, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # Your code here
        super(MultiHeadedAttention, self).__init__()
        # check if d_model is divisible by h
        assert d_model % h == 0 
        
        # num attenton heads
        self.h = h
        # dimension of each head
        self.d_k = d_model // h  
        
        # linear layers for projecting query, key, and value to multi-head 
        self.linear_q = nn.Linear(d_model, d_model)  # query
        self.linear_k = nn.Linear(d_model, d_model)  # key
        self.linear_v = nn.Linear(d_model, d_model)  # val
        
        # linear layer to combine all heads
        self.linear_out = nn.Linear(d_model, d_model)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # attention weights
        self.attn = None

    def forward(self, query, key, value, mask=None):
        # Your code here
        batch_size = query.size(0)
        
        # linear projections
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # apply attention
        out, self.attn = attention(query, key, value, mask, self.dropout)     
        
        # concat attention heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        # last linear transformation to combine heads
        out = self.linear_out(out)
        
        return out
    
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

    

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())    

    