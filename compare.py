"""
Compare standard GPT vs ETHOS architectures.
Shows parameter counts, theoretical compute, and memory usage.
"""

import sys
sys.path.append('./ethos')

from model import GPT, GPTConfig
from model_ethos import GPTETHOS, GPTConfigETHOS

def format_number(num):
    """Format large numbers nicely"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)

def count_parameters(model):
    """Count total and active parameters"""
    total = sum(p.numel() for p in model.parameters())
    return total

def theoretical_flops(config, is_moe=False):
    """Estimate FLOPs per token"""
    L = config.n_layer
    d = config.n_embd
    
    # Attention FLOPs (same for both)
    attn_flops = L * (4 * d * d + 2 * config.block_size * d)
    
    # FFN FLOPs
    if is_moe:
        # Only active experts
        moe_layers = L - getattr(config, 'num_dense_layers', 0)
        dense_layers = L - moe_layers
        
        # Dense layers
        ffn_flops = dense_layers * 8 * d * d
        
        # MoE layers (routing + active experts)
        routing_flops = moe_layers * config.num_routing_heads * d * config.d_query
        expert_flops = moe_layers * config.num_routing_heads * config.top_k * 4 * d
        ffn_flops += routing_flops + expert_flops
    else:
        # All dense
        ffn_flops = L * 8 * d * d
    
    return attn_flops + ffn_flops

def compare_architectures():
    """Compare standard GPT and ETHOS architectures"""
    
    print("Architecture Comparison: Standard GPT vs ETHOS MoE")
    print("=" * 70)
    
    # Configuration
    base_config = {
        'n_layer': 24,
        'n_head': 16,
        'n_embd': 1024,
        'block_size': 1024,
        'vocab_size': 50257,
        'dropout': 0.0,
        'bias': False,
    }
    
    # Create standard GPT
    config_standard = GPTConfig(**base_config)
    model_standard = GPT(config_standard)
    params_standard = count_parameters(model_standard)
    
    # Create ETHOS GPT
    ethos_config = {
        **base_config,
        'use_moe': True,
        'num_dense_layers': 4,
        'num_experts': 262144,  # 512^2
        'd_latent': 128,
        'd_intermediate_hypernet': 512,
        'top_k': 16,
        'num_routing_heads': 8,
        'd_query': 512,
        'use_triton': False,
    }
    config_ethos = GPTConfigETHOS(**ethos_config)
    model_ethos = GPTETHOS(config_ethos)
    params_ethos = count_parameters(model_ethos)
    
    # Calculate metrics
    flops_standard = theoretical_flops(config_standard, is_moe=False)
    flops_ethos = theoretical_flops(config_ethos, is_moe=True)
    
    # Theoretical capacity (if all experts were expanded)
    total_experts = (config_ethos.n_layer - config_ethos.num_dense_layers) * config_ethos.num_experts
    params_per_expert = 2 * config_ethos.n_embd  # Approximation
    theoretical_capacity = params_ethos + total_experts * params_per_expert
    
    # Display results
    print(f"\nModel Configuration:")
    print(f"  Layers: {base_config['n_layer']} ({config_ethos.num_dense_layers} dense + {base_config['n_layer']-config_ethos.num_dense_layers} MoE)")
    print(f"  Hidden size: {base_config['n_embd']}")
    print(f"  Attention heads: {base_config['n_head']}")
    
    print(f"\nETHOS Configuration:")
    print(f"  Experts per layer: {format_number(config_ethos.num_experts)}")
    print(f"  Active experts (top-k): {config_ethos.top_k}")
    print(f"  Routing heads: {config_ethos.num_routing_heads}")
    print(f"  Latent dimension: {config_ethos.d_latent}")
    
    print(f"\nParameter Comparison:")
    print(f"  Standard GPT: {format_number(params_standard)}")
    print(f"  ETHOS GPT: {format_number(params_ethos)}")
    print(f"  Parameter increase: {params_ethos/params_standard:.2f}x")
    print(f"  Theoretical capacity: {format_number(theoretical_capacity)} (if all experts expanded)")
    
    print(f"\nCompute Comparison (FLOPs per token):")
    print(f"  Standard GPT: {format_number(flops_standard)}")
    print(f"  ETHOS GPT: {format_number(flops_ethos)}")
    print(f"  Compute ratio: {flops_ethos/flops_standard:.2f}x")
    
    print(f"\nMemory Efficiency:")
    print(f"  Expert compression ratio: {theoretical_capacity/params_ethos:.1f}x")
    print(f"  Active parameters per token: ~{format_number(flops_ethos/2)} (approximate)")
    
    print(f"\nKey Advantages:")
    print(f"  ✓ {format_number(config_ethos.num_experts * (config_ethos.n_layer - config_ethos.num_dense_layers))} total experts")
    print(f"  ✓ {(1 - flops_ethos/flops_standard)*100:.1f}% compute reduction vs dense")
    print(f"  ✓ {config_ethos.num_routing_heads}x routing diversity")
    print(f"  ✓ O(√N) routing complexity")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    compare_architectures()