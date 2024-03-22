import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    dropout: float = 0.0
    bias: bool = False
    
c = Config()

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, block_size = 1024):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.hidden = nn.Linear(hidden_size, hidden_size * 3, bias=c.bias) # q, k, v
        
        self.attn_dropout = nn.Dropout(c.dropout)
        self.proj_dropout = nn.Dropout(c.dropout)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("No torch.nn.functional.scaled_dot_product_attention")
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
            
    def forward(self, x):
        batch_size, seq_len, hideen = x.size()
        q, k, v  = self.hidden(x).split(self.hidden_size, dim=2)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        if not self.flash:
            scores = scores =  (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            att = scores.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            att = F.softmax(scores, dim=-1)
            att = self.attn_dropout(att)
            context = att @ v
        else:
            context = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.1 if self.training else 0, is_causal=True)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hideen)
        return self.proj_dropout(self.out(context))

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, bias=False):
        super(FeedForward, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=bias),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.nn(x)
    
class Block(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Block, self).__init__()
        self.ln1 = LayerNorm(hidden_size, bias=c.bias)
        self.ln2 = LayerNorm(hidden_size, bias=c.bias)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        # x = nn.Dropout(c.dropout)(x)
        return x
        
class AGI(nn.Module):
    def __init__(self, config):
        super(AGI, self).__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.block_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config.hidden_size, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.hidden_size, bias=config.bias)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.config = config
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
    
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        assert seq_len <= self.block_size, "Cannot forward, model block size is exhausted."
        te = self.tok_emb(idx)
        pe = self.pos_emb(torch.arange(seq_len, device=idx.device).unsqueeze(0))
        x = self.drop(te + pe)
        x = self.blocks(x)
        x = self.ln_f(x)
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            logits = self.head(x[:, [-1], :]) # only the last token, see karpathy minGPT
            loss = None
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None, num_samples=1):
        for _ in range(n):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')
                
            p = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(p, num_samples=num_samples)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx