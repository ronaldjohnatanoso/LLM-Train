# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

device = 'cuda'
compile=False

init_from='scratch'

wandb_log = False
wandb_project = 'convers_slide'
wandb_run_name='slide-32w-8'

out_dir='out-convers-slide'
# saves the model if its good enough
eval_interval = 100//2 # keep frequent because we'll overfit, orig 250
# how may batches to do for evaluation 
eval_iters = 200//2
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12//2
block_size = 1024//2
#orig 5
gradient_accumulation_steps = 1 * 8

n_layer = 12//2
n_head = 12

dataset = 'conversations'
# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000


# weight decay
weight_decay = 1e-1
