import torch
stats = torch.load("pretrained/rae_dinov2_base-latents-stats.pt", map_location='cpu')
print("mean:", stats['mean'])
print("var:", stats['var'])
print("mean type:", type(stats['mean']))
print("mean shape:", stats['mean'].shape if hasattr(stats['mean'], 'shape') else "无shape")