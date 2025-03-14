# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

device = 'cuda'
compile=True

init_from='resume'

if init_from == 'resume':
    # change this manually to the our dir 
    out_dir = 'out-local-w128-54M-r8'

is_test = False
# full, slide, local
model_type = 'local' 

#run number counter, increment after each run, used for wandb run name
# dont increment for test runs
run_number = {
    'full': 5,
    'slide': 1,
    'local': 8
} 

#256, 512, 128
window_size = 128

wandb_log = True
wandb_project = f'{model_type}-conversation'
# wandb_run_name=f'{model_type}-{window_size}w-{run_number[model_type]}'
# wandb_run_id = f'{model_type}-{window_size}w-{run_number[model_type]}'

if model_type == 'full':
    window_size = 'full'



if is_test:
    wandb_log = False
    out_dir=f'out-conversation-{model_type}-w{window_size}-r{run_number[model_type]}' + '-test'
    
# saves the model if its good enough
eval_interval = 100//1 # keep frequent because we'll overfit, orig 250
# how may batches to do for evaluation 
eval_iters = 200//1
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12//2
block_size = 1024//1
#orig 5
gradient_accumulation_steps = 2* 8

n_layer = 12//6
n_head = 12//6

dataset = 'conversations'
# this makes total number of tokens be 300B
max_iters = 100000
lr_decay_iters = 600000


# weight decay
weight_decay = 1e-1


