# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

device = 'cuda'
compile=False

wandb_log = True
wandb_project = 'test-shakespeare-gpt2-smallest'
wandb_run_name='w-6'

out_dir='out-test'
# saves the model if its good enough
eval_interval = 250 # keep frequent because we'll overfit
# how may batches to do for evaluation 
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12//2
block_size = 1024//2
#orig 5
gradient_accumulation_steps = 5 * 8

dataset = 'testshakespeare'
# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000


# weight decay
weight_decay = 1e-1
