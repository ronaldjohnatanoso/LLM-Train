"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect tmux a -t pend NCCL_IB_DISABLE=1)
"""
import torch.cuda as cuda
import os
import time
import math
import pickle
from contextlib import nullcontext
import time
from datetime import timedelta
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import sys
import signal
import gc
import traceback
import atexit
import json
from model_full import GPTConfig, GPT

# set this to false if you wnat to use slide
# torch.backends.cuda.enable_cudnn_sdp(True)

torch.backends.cuda.enable_flash_sdp(True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["NCCL_DEBUG"] = "INFO"

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
model_type = 'full'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_run_id = 'default'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
window_size = 32
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
model_params = 0 # model params
run_number = {
    'full': 0,
    'slide': 0,
    'local': 0
}
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# override the model type to use the correct model file
if model_type == 'full':
    from model_full import GPT, GPTConfig
elif model_type == 'slide':
    from model_slide import GPT, GPTConfig
elif model_type == 'local':
    from model_local import GPT, GPTConfig


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend, timeout=timedelta(minutes=5))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}", flush=True)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})", flush=True)

# model init that uses window_size if needed
if model_type == 'slide' or model_type == 'local':
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, window_size=window_size) # start with model_args from command line
# model does not use window_size
else:
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line




if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch", flush=True)
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)", flush=True)
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}", flush=True)
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# elif init_from.startswith('gpt2'):
#     print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#     # initialize from OpenAI GPT-2 weights
#     override_args = dict(dropout=dropout)
#     model = GPT.from_pretrained(init_from, override_args)
#     # read off the created config params, so we can store them into checkpoint correctly
#     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#         model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)



# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)", flush=True)
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


#model params
model_params =f"{round(sum(p.numel() for p in model.parameters()) / 1e6)}M" 


#create the out_dir with model_params appended name
wandb_run_name=f'{model_type}-w{window_size}-{model_params}-r{run_number[model_type]}'
wandb_run_id = f'{model_type}-w{window_size}-{model_params}-r{run_number[model_type]}'
out_dir = f'out-{model_type}-w{window_size}-{model_params}-r{run_number[model_type]}'
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# logging
if wandb_log and master_process:


    import wandb
    if init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, resume='must', id=wandb_run_id)
    else:
        print('your wandb run id is:', wandb_run_id)
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=wandb_run_id)

# Define a cleanup function
def cleanup():
    """Clean up resources properly before exit."""
    if torch.cuda.is_available():
        log_with_flush("Cleaning up CUDA memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        log_with_flush("CUDA memory cleanup complete")

# Register cleanup function to be called on normal program termination
atexit.register(cleanup)

# Set up signal handlers for graceful termination
def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    log_with_flush(f"\nReceived signal {sig}, terminating gracefully...")
    cleanup()
    if ddp:
        log_with_flush("Destroying process group...")
        destroy_process_group()
    log_with_flush("Training terminated by signal")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Termination signal

# add a helper function for consistent logging
def log_with_flush(message):
    """Log a message and flush stdout to ensure it's displayed immediately."""
    print(message, flush=True)
    sys.stdout.flush()  # Double ensure flushing

# Add a wrapper for the main training loop with error handling
def run_training():
    global iter_num, local_iter_num, best_val_loss, running_mfu, last_log_time, timestamp_checkpoint
    
    X, Y = get_batch('train')  # fetch the very first batch
    t0 = time.time()
    
    # Our main loop should be protected with a try/except
    try:
        while True:
            # Memory check at the start of each iteration
            if iter_num % log_interval == 0 and master_process:
                try:
                    torch.cuda.synchronize()
                    log_with_flush(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                                   f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                except Exception as e:
                    log_with_flush(f"Error checking GPU memory: {str(e)}")

            # Periodic heartbeat log to confirm the training is still running
            current_time = time.time()
            if current_time - last_log_time > 600:  # Log at least every 10 minutes
                if master_process:
                    log_with_flush(f"Heartbeat - Training still in progress at iteration {iter_num}, time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                last_log_time = current_time

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and master_process:
                # save the iter number and steps in a json inside its out_dir
                with open(os.path.join(out_dir, 'steps.json'), 'w') as f:
                    json.dump({'iter_num': iter_num, 'steps' : iter_num//eval_interval}, f, indent=4)
                
                try:
                    losses = estimate_loss()
                    log_message = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    log_with_flush(log_message)
                    
                    if wandb_log:
                        wandb.log({
                            "iter": iter_num,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "lr": lr,
                            "mfu": running_mfu*100, # convert to percentage
                        })
                    if losses['val'] < best_val_loss or always_save_checkpoint:
                        best_val_loss = losses['val']
                        if iter_num > 0:
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'model_args': model_args,
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': config,
                            }
                            log_with_flush(f"saving checkpoint to {out_dir}")
                            
                            # Save with try/except to handle potential disk issues
                            try:
                                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                            except Exception as e:
                                log_with_flush(f"Error saving checkpoint: {str(e)}")
                                log_with_flush(traceback.format_exc())
                except Exception as e:
                    log_with_flush(f"Error during evaluation: {str(e)}")
                    log_with_flush(traceback.format_exc())
                    
            if iter_num == 0 and eval_only:
                break

            # GPU memory monitoring
            if iter_num % log_interval == 0 and master_process and device_type == 'cuda':
                try:
                    # Ensure all previous CUDA work is completed
                    cuda.synchronize()
                        
                    # Get memory statistics
                    total_memory = cuda.get_device_properties(0).total_memory
                    reserved_memory = cuda.memory_reserved(0)
                    allocated_memory = cuda.memory_allocated(0)
                    free_memory = total_memory - reserved_memory
                        
                    memory_log = [
                        f"GPU Memory before iteration {iter_num}:",
                        f"  Total: {total_memory / 1e9:.2f} GB",
                        f"  Reserved: {reserved_memory / 1e9:.2f} GB",
                        f"  Allocated: {allocated_memory / 1e9:.2f} GB",
                        f"  Free: {free_memory / 1e9:.2f} GB",
                    ]
                    
                    # Estimate memory needed for this iteration
                    batch_size_total = batch_size * gradient_accumulation_steps
                    estimated_memory = (
                        batch_size_total * block_size * model.config.n_embd * 4 +  # Input tensors
                        batch_size_total * block_size * model.config.vocab_size * 4 +  # Output logits
                        sum(p.numel() * 4 for p in model.parameters())  # Model parameters
                    )
                    memory_log.append(f"  Estimated memory for this iteration: {estimated_memory / 1e9:.2f} GB")
                    
                    log_with_flush("\n".join(memory_log))
                except Exception as e:
                    log_with_flush(f"Error during memory monitoring: {str(e)}")

            # Training step with error handling
            try:
                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                for micro_step in range(gradient_accumulation_steps):
                    if ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                    with ctx:
                        logits, loss = model(X, Y)
                        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    X, Y = get_batch('train')
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()
                # clip the gradient
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                
                # Explicit garbage collection every few iterations
                if iter_num % 100 == 0:
                    gc.collect()
                    if device_type == 'cuda':
                        torch.cuda.empty_cache()
            except Exception as e:
                log_with_flush(f"Error during training step (iteration {iter_num}):")
                log_with_flush(traceback.format_exc())
                
                # Try to recover from OOM or other errors
                if device_type == 'cuda':
                    log_with_flush("Attempting to recover by clearing GPU memory cache...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                # Try to get a new batch and continue
                try:
                    log_with_flush("Attempting to fetch a new batch and continue...")
                    X, Y = get_batch('train')
                    log_with_flush("Successfully fetched a new batch, continuing training")
                    # If we've had multiple consecutive errors, we might want to exit
                except Exception as e2:
                    log_with_flush(f"Failed to recover from error: {str(e2)}")
                    log_with_flush("Saving emergency checkpoint and exiting...")
                    
                    # Try to save an emergency checkpoint
                    try:
                        emergency_checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        torch.save(emergency_checkpoint, os.path.join(out_dir, f'emergency_ckpt_{iter_num}.pt'))
                        log_with_flush(f"Emergency checkpoint saved to {out_dir}/emergency_ckpt_{iter_num}.pt")
                    except Exception as e3:
                        log_with_flush(f"Failed to save emergency checkpoint: {str(e3)}")
                    
                    cleanup()
                    if ddp:
                        destroy_process_group()
                    return

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                log_with_flush(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
                
                current_time=time.time()
                time_since_last_checkpoint = current_time - timestamp_checkpoint
                timestamp_checkpoint = current_time
                # Convert time since last checkpoint to a formatted string
                formatted_time_since_last_checkpoint = str(timedelta(seconds=int(time_since_last_checkpoint)))

                log_with_flush(f"Time stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}, Time since last checkpoint: {formatted_time_since_last_checkpoint}")
                last_log_time = current_time  # Reset the heartbeat timer when we log normally
           
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                log_with_flush(f"Training complete after {iter_num} iterations")
                break
    
    except KeyboardInterrupt:
        log_with_flush("Interrupted by user. Cleaning up...")
        cleanup()
    except Exception as e:
        log_with_flush(f"Unhandled exception: {str(e)}")
        log_with_flush(traceback.format_exc())
        cleanup()
        # Save emergency checkpoint if possible
        try:
            emergency_checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(emergency_checkpoint, os.path.join(out_dir, f'emergency_ckpt_{iter_num}.pt'))
            log_with_flush(f"Emergency checkpoint saved to {out_dir}/emergency_ckpt_{iter_num}.pt")
        except Exception as e2:
            log_with_flush(f"Failed to save emergency checkpoint: {str(e2)}")
        
        if ddp:
            destroy_process_group()
        return
    
    log_with_flush("Training completed successfully!")
    return


def save_important_vars(model):
    # Define the important variables
    important_vars = {
        "num_parameters": model_params,
        "out_dir": out_dir,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "eval_iters": eval_iters,
        "eval_only": eval_only,
        "always_save_checkpoint": always_save_checkpoint,
        "init_from": init_from,
        "model_type": model_type,
        "wandb_log": wandb_log,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_run_id": wandb_run_id,
        "dataset": dataset,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "window_size": window_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "bias": bias,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "grad_clip": grad_clip,
        "decay_lr": decay_lr,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "min_lr": min_lr,
        "backend": backend,
        "device": device,
        "dtype": dtype,

    }

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Write the important variables to a JSON file
    with open(os.path.join(out_dir, 'hyperparams.json'), 'w') as f:
        json.dump(important_vars, f, indent=4)

# Initialize variables
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
timestamp_checkpoint = time.time()
last_log_time = time.time()  # Track time of last log message

try:
    # Run the main training loop
    save_important_vars(model)
    run_training()
finally:
    # Always perform cleanup on exit
    cleanup()
    if ddp:
        destroy_process_group()
    log_with_flush("Script execution completed")



