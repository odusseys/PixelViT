import numpy as np
import torch
from PIL import Image
from IPython.display import display
import torch.nn as nn
import torchvision

locations_cache = {}

def make_base_locations(batch_size, size):
  key = f"{batch_size}_{size}"
  if key not in locations_cache:
    res = np.zeros((size, size, 2))
    for i in range(size):
      for j in range(size):
        res[i, j, 0] = 2 * j / size - 1
        res[i, j, 1] = 2 * i / size - 1
    locations_cache[key] = torch.tensor(res, requires_grad=False, dtype=torch.float32,
                        device="cuda").unsqueeze(0).repeat(batch_size, 1, 1, 1)
  return locations_cache[key]

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def fit(self, X):
        b, n, d = X.shape
        self.register_buffer("mean_", X.mean(1, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh.transpose(1, 2)[:, :, :self.n_components]
        self.register_buffer("components_", Vt)
        std = S[:, :self.n_components].unsqueeze(1).sqrt()
        self.register_buffer("std_", std)
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        unscaled = torch.bmm(X - self.mean_, self.components_)
        scaled = unscaled / self.std_  # Scale for unit variance
        return scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        Y = Y * self.std_  # Unscale
        return  torch.bmm(Y, self.components_.transpose(1, 2)) + self.mean_


def pca(f1, n):
  batch_size, n_features, _, f1_size = f1.shape
  f1 = f1.reshape((batch_size, n_features, f1_size * f1_size)).transpose(1, 2)
  pca = PCA(n)
  f1_reduced = pca.fit_transform(f1)
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  return f1_reduced, pca

def reduce_features(f1, f2, n):
  f1_size = f1.shape[-1]
  f2_size = f2.shape[-1]
  n_features = f1.shape[1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, n_features, f1_size * f1_size)).transpose(1, 2)
  f2 = f2.reshape((batch_size, n_features, f2_size * f2_size)).transpose(1, 2)
  pca = PCA(n)
  f1_reduced = pca.fit_transform(f1)
  f2_reduced = pca.transform(f2)
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  f2_reduced = f2_reduced.transpose(1, 2).reshape((batch_size, n, f2_size, f2_size))
  return f1_reduced, f2_reduced, pca

def reduce_dimension(f1, other_features, n_features, n):
  batch_size = f1.shape[0]
  size_1 = f1.shape[2]
  other_sizes = [f2.shape[2] for f2 in other_features]
  f1 = f1.permute((0, 2, 3, 1)).reshape((batch_size, size_1 * size_1, n_features))
  other_features = [f2.permute((0, 2, 3, 1)).reshape((batch_size, f2.shape[2] * f2.shape[2], n_features)) for f2 in other_features]
  pca = PCA(n_components=n).fit(f1)
  f1 = pca.transform(f1).reshape((batch_size, size_1, size_1, n)).permute(0, 3, 1, 2)
  other_features = [pca.transform(f2).reshape((batch_size, size_2, size_2, n)).permute(0, 3, 1, 2) for f2, size_2 in zip(other_features, other_sizes)]
  m = min(torch.min(f1), *[torch.min(f2) for f2 in other_features])
  M = max(torch.max(f1), *[torch.max(f2) for f2 in other_features])
  f1 = (f1 - m) / (M - m)
  other_features = [(f2 - m) / (M - m) for f2 in other_features]
  return f1, other_features

def debug_features(f1, other_features, display_size=128):
  n_features = f1.shape[0]
  f1, other_features = reduce_dimension(f1.unsqueeze(0), [f2.unsqueeze(0) for f2 in other_features], n_features, 3)
  f1 = f1[0].permute(1, 2, 0).detach().cpu().numpy().squeeze() * 255
  images = [Image.fromarray(f1.astype(np.uint8)).resize((display_size, display_size), 0)]
  for f2 in other_features:
    f2 = f2[0].permute(1, 2, 0).detach().cpu().numpy().squeeze() * 255
    images.append(Image.fromarray(f2.astype(np.uint8)).resize((display_size, display_size), 0))
  return Image.fromarray(np.hstack(images).astype(np.uint8))

def debug_individual_features(features, size=128, clip=5):
  images = []
  for i in range(min(5, len(features))):
    f = np.clip(features[i], -clip, clip)
    f = (f + clip) / (2 * clip) * 255
    images.append(Image.fromarray(f.astype(np.uint8)).resize((size, size), 0))
  # make row of images and display
  display(Image.fromarray(np.hstack(images).astype(np.uint8)))


N_SEGMENTATION_CLASSES = 133

def get_segmentation_map(image, mask):
  image = (image * 255).to(dtype=torch.uint8)

  # make boolean segmentation masks
  mask = torch.nn.functional.interpolate(
      mask.unsqueeze(0),
      size=image.shape[-1],
      mode="nearest",
  ).squeeze(1).to(dtype=torch.long)
  one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=N_SEGMENTATION_CLASSES + 1).squeeze()
  one_hot_mask = one_hot_mask.permute(2, 0, 1).contiguous()
  boolean_mask = one_hot_mask.type(torch.bool)
  res = torchvision.utils.draw_segmentation_masks(image, boolean_mask).permute(1, 2, 0)
  return Image.fromarray(res.cpu().numpy())