import correlation
import torch

B, C, H, W = 1, 1, 32, 32

a = torch.randint(1, 4, (B, C, H, W), dtype=torch.float32).cuda()
b = torch.randint_like(a, 1, 4).cuda()

print(a.dtype)
print(a.shape, b.shape)
print(a.device, b.device)

corr = correlation.Correlation(pad_size=4,
                               kernel_size=1,
                               max_displacement=4,
                               stride1=1,
                               stride2=1,
                               corr_multiply=1).cuda()

c = corr(a, b)
print(c.shape)
