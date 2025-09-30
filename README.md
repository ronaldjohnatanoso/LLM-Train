# LLM-Train

A PyTorch-based training framework for GPT-style language models with multiple architecture variants and comprehensive training utilities.

## Overview

LLM-Train is a flexible training framework that supports three different GPT model architectures:
- **Full Model** (`model_full.py`) - Complete GPT implementation with full attention
- **Slide Model** (`model_slide.py`) - Sliding window attention variant
- **Local Model** (`model_local.py`) - Local attention implementation

The framework supports distributed training, various datasets, and includes comprehensive monitoring and sampling capabilities.

## Features

- 🚀 **Multiple Model Architectures**: Choose from full, sliding window, or local attention models
- 🔧 **Flexible Configuration**: Easy-to-use config system for hyperparameter tuning
- 📊 **Weights & Biases Integration**: Built-in logging and experiment tracking
- 🎯 **Multiple Datasets**: Support for conversations, Shakespeare, OpenWebText, and custom datasets
- 🎲 **Text Generation**: Comprehensive sampling utilities with temperature and top-k controls
- 📈 **Performance Monitoring**: Real-time training monitoring and benchmarking tools

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM-Train
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify CUDA installation (optional):
```bash
python cuda_check.py
```

## Quick Start

### 1. Prepare Your Data

For conversation data:
```bash
# Place your CSV file in data/conversations/ as conv.csv
cd data/conversations/
python csv_prep.py  # Converts CSV to input.txt
python prepare.py   # Creates train.bin and val.bin
```

For other datasets, check the respective folders in `data/`:
- `data/shakespeare/` - Shakespeare text
- `data/openwebtext/` - OpenWebText dataset
- `data/shakespeare_char/` - Character-level Shakespeare

### 2. Configure Training

Edit `config/train_conversation.py` or create your own config:

```python
# Model selection
model_type = 'slide'  # 'full', 'slide', or 'local'

# Training parameters
batch_size = 6
block_size = 1024
n_layer = 12
n_head = 12
window_size = 16  # For slide/local models

# Logging
wandb_log = True
wandb_project = 'my-llm-experiment'
```

### 3. Start Training

```bash
python train.py config/train_conversation.py --device=cuda --out_dir=my-model-run
```

For distributed training:
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_conversation.py
```

### 4. Generate Text

```bash
python sample.py \
    --out_dir=my-model-run \
    --start="What is the meaning of life?" \
    --num_samples=5 \
    --max_new_tokens=500
```

## Model Architectures

### Full Model (`model_full.py`)
- Complete GPT-2 style transformer
- Full attention mechanism
- Best for smaller sequences and maximum quality

### Slide Model (`model_slide.py`)
- Sliding window attention
- Configurable window size
- Good balance of performance and memory efficiency

### Local Model (`model_local.py`)
- Local attention patterns
- Memory efficient for long sequences
- Faster training on large contexts

## Configuration System

The framework uses a hierarchical configuration system:

1. **Default values** in `train.py`
2. **Config files** in `config/` directory override defaults
3. **Command line arguments** override config files

### Key Configuration Files

- `config/train_conversation.py` - Conversation training setup
- `config/train_gpt2.py` - GPT-2 replication setup
- `config/finetune_shakespeare.py` - Shakespeare fine-tuning

### Important Parameters

```python
# Model architecture
model_type = 'slide'        # Model variant
n_layer = 12               # Number of transformer layers
n_head = 12                # Number of attention heads
n_embd = 768               # Embedding dimension
window_size = 16           # Attention window (slide/local only)

# Training
batch_size = 6             # Batch size per GPU
block_size = 1024          # Sequence length
learning_rate = 6e-4       # Peak learning rate
max_iters = 100000         # Training iterations
gradient_accumulation_steps = 8  # Gradient accumulation

# Evaluation
eval_interval = 100        # Evaluation frequency
eval_iters = 200          # Evaluation iterations
always_save_checkpoint = True  # Save all checkpoints
```

## Data Preparation

### Conversation Data
1. Place CSV file as `data/conversations/conv.csv`
2. CSV should have format: `[column_a, question, answer]`
3. Run `csv_prep.py` to extract questions and answers
4. Run `prepare.py` to tokenize and create binary files

### Custom Data
1. Create a folder in `data/your_dataset/`
2. Add your text data as `input.txt`
3. Create a `prepare.py` script following existing examples
4. Update dataset name in your config file

## Training Commands

### Basic Training
```bash
python train.py config/train_conversation.py
```

### With Custom Parameters
```bash
python train.py config/train_conversation.py \
    --batch_size=8 \
    --learning_rate=3e-4 \
    --max_iters=50000 \
    --out_dir=custom-run
```

### Resume Training
```bash
python train.py config/train_conversation.py \
    --init_from=resume \
    --out_dir=existing-run
```

### Distributed Training
```bash
# Single node, multiple GPUs
torchrun --standalone --nproc_per_node=4 train.py config/train_conversation.py

# Multiple nodes
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=1234 \
    train.py config/train_conversation.py
```

## Text Generation

### Basic Sampling
```bash
python sample.py --out_dir=my-model --start="Hello world"
```

### Advanced Sampling
```bash
python sample.py \
    --out_dir=my-model \
    --start="Write a story about" \
    --num_samples=10 \
    --max_new_tokens=1000 \
    --temperature=0.8 \
    --top_k=200 \
    --seed=42
```

### Sampling Parameters
- `temperature`: Controls randomness (0.1 = conservative, 1.0 = balanced, 2.0 = creative)
- `top_k`: Only consider top-k most likely tokens
- `max_new_tokens`: Maximum tokens to generate
- `num_samples`: Number of independent samples

## Monitoring and Logging

### Weights & Biases
Enable W&B logging in your config:
```python
wandb_log = True
wandb_project = 'my-project'
wandb_run_name = 'experiment-1'
```

### Training Monitoring
```bash
python monitor_training.py --out_dir=my-model
```

### Benchmarking
```bash
python bench.py --batch_size=12 --compile=True
```

## Project Structure

```
LLM-Train/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train.py                 # Main training script
├── sample.py                # Text generation script
├── model_full.py            # Full attention model
├── model_slide.py           # Sliding window model
├── model_local.py           # Local attention model
├── config/                  # Configuration files
│   ├── train_conversation.py
│   ├── train_gpt2.py
│   └── ...
├── data/                    # Dataset directories
│   ├── conversations/
│   ├── shakespeare/
│   ├── openwebtext/
│   └── ...
├── out/                     # Training outputs
└── assets/                  # Images and resources
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `batch_size` or `block_size`
- Enable gradient checkpointing
- Use smaller model (`n_layer`, `n_head`, `n_embd`)

**Slow Training**
- Enable compilation: `compile=True`
- Use mixed precision: `dtype='bfloat16'`
- Increase `gradient_accumulation_steps`

**Poor Generation Quality**
- Train longer (`max_iters`)
- Adjust learning rate schedule
- Try different model architectures
- Increase model size

**Data Loading Errors**
- Ensure `train.bin` and `val.bin` exist in dataset folder
- Check dataset path in config
- Verify data preparation scripts ran successfully

### Performance Tips

1. **Use Flash Attention**: Enabled by default in PyTorch 2.0+
2. **Compile Models**: Set `compile=True` for 20-30% speedup
3. **Mixed Precision**: Use `bfloat16` on modern GPUs
4. **Gradient Accumulation**: Increase for larger effective batch sizes
5. **Window Size**: Tune for slide/local models based on your data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Based on Andrej Karpathy's nanoGPT
- Inspired by OpenAI's GPT-2 architecture
- Uses Hugging Face tokenizers and datasets
