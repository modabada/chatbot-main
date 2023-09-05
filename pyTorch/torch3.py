import torch

a = torch.tensor(
    [[1, 2, 3, 4], [3, 4, 5, 6]], 
    dtype=torch.float64,
    # device='cuda:0'       WARNING: CUDA not configured!
)
b = a + 3
print(a.cpu().numpy())
print(a[0][0])
print(a[1][0])
print(a[1, 0])

# slicing
print(a[0, 1:3])
print(b)

# show size
print(a.shape)

# calculate tensor to tensor
c = a + b
print(c)

# change tensor dimension
print(c.view(8, 1))
print(c.view(1, 8))
print(c.view(2, 2, 2))
print(c.view(2, -1, 2).shape)