import torch
import torch.nn as nn

# https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/nhwc.html

# memory_format = torch.channels_last
memory_format = torch.contiguous_format
device = "cuda"
n_iters = 1000
batch_size = 8

model = nn.Conv2d(32, 64, 3, 1, 1).to(device=device)
x = torch.randn(batch_size, 32, 128, 128).to(device=device)
x = x.contiguous(memory_format=memory_format)
x = x.to(memory_format=memory_format)

print(x.is_contiguous(memory_format=memory_format))
print(x.shape)

model = model.to(memory_format=memory_format)

with torch.autograd.profiler.profile(use_device=device) as prof:
    for _ in range(n_iters):
        y: torch.Tensor = model(x)
   
print(prof.key_averages().table(sort_by="self_device_time_total"))

print(y.shape)
print(y.is_contiguous(memory_format=memory_format))
