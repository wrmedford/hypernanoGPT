"""
Compare standard FFN vs hypernetwork FFN
Using much smaller hypernetwork dimensions for actual compression
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTConfig, GPT
from contextlib import nullcontext

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

class HypernetFFN(nn.Module):
    """Hypernetwork FFN with aggressive compression settings"""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)
        
        # Use very small hypernetwork for actual compression
        self.d_latent = 8  # Very small latent
        self.d_hidden = 16  # Very small hidden
        self.rank = 4  # Very low rank
        
        # Latent code
        self.latent = nn.Parameter(torch.randn(self.d_latent) * 0.02)
        
        # Tiny hypernetwork
        self.hyper1 = nn.Linear(self.d_latent, self.d_hidden, bias=False)
        self.hyper2 = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
        
        # Generate only low-rank factors
        # Total params for factors: d_hidden * (n_embd + 4*n_embd) * rank * 2
        self.to_fc_factors = nn.Linear(self.d_hidden, (self.n_embd + 4 * self.n_embd) * self.rank, bias=False)
        self.to_proj_factors = nn.Linear(self.d_hidden, (4 * self.n_embd + self.n_embd) * self.rank, bias=False)
        
        # Initialize small to prevent explosion
        with torch.no_grad():
            self.hyper1.weight.mul_(0.1)
            self.hyper2.weight.mul_(0.1)
            self.to_fc_factors.weight.mul_(0.1)
            self.to_proj_factors.weight.mul_(0.1)
        
    def forward(self, x):
        # Generate hidden state
        h = F.gelu(self.hyper1(self.latent))
        h = self.hyper2(h)
        
        # Generate factors for c_fc
        fc_factors = self.to_fc_factors(h)
        fc_u, fc_v = fc_factors.split([self.n_embd * self.rank, 4 * self.n_embd * self.rank])
        fc_u = fc_u.view(self.n_embd, self.rank)
        fc_v = fc_v.view(self.rank, 4 * self.n_embd)
        
        # Generate factors for c_proj  
        proj_factors = self.to_proj_factors(h)
        proj_u, proj_v = proj_factors.split([4 * self.n_embd * self.rank, self.n_embd * self.rank])
        proj_u = proj_u.view(4 * self.n_embd, self.rank)
        proj_v = proj_v.view(self.rank, self.n_embd)
        
        # Apply FFN with generated weights
        x = F.linear(x, (fc_u @ fc_v).t())
        x = F.gelu(x)
        x = F.linear(x, (proj_u @ proj_v).t())
        x = self.dropout(x)
        return x

# Even simpler version - just directly parameterize low-rank factors
class SimpleHypernetFFN(nn.Module):
    """Simplest possible compression - just low-rank parameterization"""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)
        self.rank = 16  # Reasonable rank
        
        # Just store low-rank factors directly
        self.fc_u = nn.Parameter(torch.randn(self.n_embd, self.rank) * 0.02)
        self.fc_v = nn.Parameter(torch.randn(self.rank, 4 * self.n_embd) * 0.02)
        self.proj_u = nn.Parameter(torch.randn(4 * self.n_embd, self.rank) * 0.02)
        self.proj_v = nn.Parameter(torch.randn(self.rank, self.n_embd) * 0.02)
        
    def forward(self, x):
        x = F.linear(x, (self.fc_u @ self.fc_v).t())
        x = F.gelu(x)
        x = F.linear(x, (self.proj_u @ self.proj_v).t())
        x = self.dropout(x)
        return x

# Test both versions
import model

# Modified Block
class ModifiedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = model.CausalSelfAttention(config)
        self.ln_2 = model.LayerNorm(config.n_embd, bias=config.bias)
        
        hypernet_type = getattr(config, 'hypernet_type', None)
        if hypernet_type == 'full':
            self.mlp = HypernetFFN(config)
        elif hypernet_type == 'simple':
            self.mlp = SimpleHypernetFFN(config)
        else:
            self.mlp = model.MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Save and replace
OriginalBlock = model.Block
model.Block = ModifiedBlock

print("Comparing three approaches:\n")

# Standard model
config_std = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=256, bias=False, vocab_size=65, dropout=0.0)
model_std = GPT(config_std).to(device)

# Simple low-rank model
config_simple = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=256, bias=False, vocab_size=65, dropout=0.0)
config_simple.hypernet_type = 'simple'
model_simple = GPT(config_simple).to(device)

# Full hypernetwork model
config_hyper = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=256, bias=False, vocab_size=65, dropout=0.0)
config_hyper.hypernet_type = 'full'
model_hyper = GPT(config_hyper).to(device)

# Count parameters
def count_ffn_params(model):
    return sum(p.numel() for name, p in model.named_parameters() if 'mlp' in name)

print("FFN Parameters (all 4 layers combined):")
print(f"Standard FFN: {count_ffn_params(model_std):,}")
print(f"Simple low-rank: {count_ffn_params(model_simple):,}")
print(f"Hypernetwork: {count_ffn_params(model_hyper):,}")

print("\nPer-layer breakdown:")
print(f"Standard: {count_ffn_params(model_std) // 4:,} params/layer")
print(f"Simple low-rank: {count_ffn_params(model_simple) // 4:,} params/layer")
print(f"Hypernetwork: {count_ffn_params(model_hyper) // 4:,} params/layer")

print("\nCompression ratios:")
print(f"Simple low-rank: {count_ffn_params(model_std) / count_ffn_params(model_simple):.2f}x")
print(f"Hypernetwork: {count_ffn_params(model_std) / count_ffn_params(model_hyper):.2f}x")

# Test forward pass
print("\nTesting forward pass...")
x = torch.randint(0, 65, (4, 256), device=device)
y = torch.randint(0, 65, (4, 256), device=device)

with ctx:
    _, loss_std = model_std(x, y)
    _, loss_simple = model_simple(x, y)
    _, loss_hyper = model_hyper(x, y)
    
print(f"Standard loss: {loss_std.item():.4f}")
print(f"Simple low-rank loss: {loss_simple.item():.4f}")
print(f"Hypernetwork loss: {loss_hyper.item():.4f}")

# Calculate what we're actually storing vs generating
print("\nWhat we're actually doing:")
print(f"Standard: Storing {count_ffn_params(model_std) // 4:,} params to get {131_072} param FFN")
print(f"Simple: Storing {count_ffn_params(model_simple) // 4:,} params to generate {131_072} param FFN")
print(f"Hypernet: Storing {count_ffn_params(model_hyper) // 4:,} params to generate {131_072} param FFN")

# Restore
model.Block = OriginalBlock
