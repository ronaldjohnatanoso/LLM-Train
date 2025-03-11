device = 'cuda'
compile = False

init_from = 'scratch'

is_test = True
# full, slide, local
model_type = 'full'

# Run number counter, increment after each run, used for wandb run name
# Don't increment for test runs
run_number = {
    'full': 0,
    'slide': 0,
    'local': 0
}

window_size = 32

wandb_log = False  # Disable for test runs
wandb_project = f'{model_type}-conversation-test'
wandb_run_name = f'{model_type}-{window_size}w-{run_number[model_type]}'
out_dir = f'out-conversation-{model_type}-test'

# Saves the model if it's good enough
eval_interval = 50  # Reduce frequency to save compute
eval_iters = 100
log_interval = 10  

always_save_checkpoint = True

# Optimize batch size and sequence length for 8GB VRAM
batch_size = 1  # Reduce from 3 to 1
block_size = 512  # Reduce from 1024 to 512
gradient_accumulation_steps = 2  # Reduce from 8 to 2

n_layer = 12  # Can reduce to 8 if OOM issues occur
n_head = 12   # Can reduce to 8 if needed

dataset = 'conversations'
max_iters = 600000  # Reduce iteration count
lr_decay_iters = 600000

weight_decay = 1e-2  # Reduce from 1e-1 to avoid over-regularization

# Enable memory-efficient techniques
mixed_precision = 'fp16'  # Enable FP16 to cut VRAM usage in half
gradient_checkpointing = True  # Saves memory at the cost of slightly slower training
