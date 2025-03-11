import torch
print(torch.cuda.is_available())  # Should return True if the GPU is usable
print(torch.cuda.get_device_name(0))  # Should return the GPU name