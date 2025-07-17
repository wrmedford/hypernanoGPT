# New config file for hypernetwork testing
out_dir = 'out-hypernet-test'
eval_interval = 100
eval_iters = 50
log_interval = 10

wandb_log = False
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# Small model for quick testing
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# Hypernetwork specific configs
use_hypernet_ffn = True
d_latent = 32
d_hidden_hyper = 64

learning_rate = 3e-4
max_iters = 1000
lr_decay_iters = 1000
min_lr = 3e-5
