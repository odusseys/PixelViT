
from pytorch_wavelets import DWTForward
import torch
from torch import nn
import torchvision
import numpy as np
import math

dwt_cache = {}

def wavelet_decomposition_torch(img, n_wavelets, size):
  def resize(x):
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-5)
    return torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=False, antialias=True)

  res = [resize(img)]
  if n_wavelets not in dwt_cache:
    xfm = DWTForward(J=n_wavelets).cuda()
    dwt_cache[n_wavelets] = xfm
  xfm = dwt_cache[n_wavelets]

  Yl, Yh = xfm(img)

  res.append(resize(Yl))
  for y in Yh:
    batch_size, n_channels, n_freq, s, s = y.shape
    y = y.reshape((batch_size, n_channels * n_freq, s, s))
    res.append(resize(y))
    del y
  return torch.cat(res, dim=1)

MINIMAL_IMAGE_SIZE = 224
MINIMAL_GRID_SIZE = MINIMAL_IMAGE_SIZE // 14

def make_positional_encodings(batch_size, size, n):
  with torch.no_grad():
    res = np.zeros((2, size, size))
    for i in range(size):
      for j in range(size):
        res[0, i, j] = math.pi * i / size
        res[1, i, j] = math.pi * j / size
    res = torch.tensor(res, requires_grad=False, dtype=torch.float32)
    res = torch.cat([res * (i + 1) for i in range(n)])
    return torch.cos(res).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda() / math.sqrt(n)

def preprocess_images(dino, images, starting_size, target_size, n_wavelets, n_pe_frequencies):
    if type(images) == list:
        images = torch.stack([torchvision.transforms.PILToTensor()(img) for img in images]).cuda() / 255.0

    factor = starting_size // MINIMAL_GRID_SIZE

    images_small = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True)

    # add noise to minimize positional encoding ViT artifacts
    images_small = (images_small + torch.randn_like(images_small) * 0.05).clamp(0, 1)

    features = dino.get_features_for_tensor(images_small)

    images = nn.functional.interpolate(images, (target_size, target_size), mode="bilinear", align_corners=False, antialias=True)
    wavelets = wavelet_decomposition_torch(images, n_wavelets, target_size)
    pe = make_positional_encodings(images.shape[0], target_size, n_pe_frequencies)
    return features, images, torch.cat([wavelets, pe], dim=1)

