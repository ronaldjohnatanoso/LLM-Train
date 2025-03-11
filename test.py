import os
import torch

# Check if the CUDNN_DISABLE environment variable is set
cudnn_disabled = os.environ.get("CUDNN_DISABLE", "1") == "0"
print(f"CUDNN_DISABLE is set: {cudnn_disabled}")

# Check if cuDNN is enabled in PyTorch
cudnn_enabled = torch.backends.cudnn.enabled
print(f"cuDNN enabled in PyTorch: {cudnn_enabled}")


print(f"PyTorch version: {torch.__version__}")

# torch.backends.cuda.enable_cudnn_sdp(True)

print('testing')