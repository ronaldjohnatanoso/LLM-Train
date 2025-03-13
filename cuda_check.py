import torch

print(torch.cuda.is_available())

# import triton
# print(triton.__version__)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'f

print(dtype)

