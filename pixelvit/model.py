import torch
import torch.nn as nn
import math
from .layers import Activation, ProjectionLayer, ConvolutionalStage
from .preprocessing import preprocess_images
from .dino import DinoV2, N_FEATURES
from .util import make_base_locations

class PixelViTStage(nn.Module):
  def __init__(self, n_features, n_features_in, n_image_features):
    super().__init__()
    self.image_encoder = ProjectionLayer(n_image_features, n_features)

    self.feature_encoder = ProjectionLayer(n_features_in, n_features)

    self.transformer = ConvolutionalStage(n_features)

    last_layer = nn.Conv2d(n_features // 8, 2, 1)

    self.deformer = nn.Sequential(
        nn.Conv2d(n_features, n_features // 2, 1),
        Activation(),
        nn.Conv2d(n_features // 2, n_features // 4, 1),
        Activation(),
        nn.Conv2d(n_features // 4, n_features // 8, 1),
        Activation(),
        last_layer,
    )

    # initialize all deformer weights to 0
    torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.01, generator=None)
    nn.init.zeros_(last_layer.bias)

    self.base_locations = None


  def forward(self, features, image):
    image = self.image_encoder(image)
    image = torch.nn.functional.interpolate(image, features.shape[-1], mode="bilinear")
    f = self.feature_encoder(features)
    f = self.transformer(f, image)
    if self.base_locations is None:
      self.base_locations = make_base_locations(image.shape[0], image.shape[-1]).permute((0, 3, 1, 2))
    field = self.base_locations + self.deformer(f)
    field = field.permute((0, 2, 3, 1))
    return dict(deformed=torch.nn.functional.grid_sample(features, field, padding_mode="border", align_corners=False), field=field)


DEFAULT_CONFIG = dict(
    n_features_in= 384,
    n_frequencies= 16,
    n_wavelets=8,
    n_intermediate_features={
        32: 256,
        64: 256,
        128: 256,
        256: 256
    },
    start_size=32,
    target_size=256,
    dino_model="s",
)

class PixelViT(nn.Module):
  def __init__(self, config):
    super().__init__()

    n_features_in = config["n_features_in"]
    n_frequencies = config["n_frequencies"]
    n_wavelets = config["n_wavelets"]
    n_intermediate_features = config["n_intermediate_features"]

    n_pe_features = 2 * n_frequencies
    n_image_features = 6 + (3 * 3 * n_wavelets) + n_pe_features

    n_upscales = int(math.log2(config["target_size"] // config["start_size"])) + 1
    assert n_upscales == len(n_intermediate_features), "Incompatible resolutions and feature config"

    self.stages = nn.ModuleList([
        PixelViTStage(n_intermediate_features[res], n_features_in, n_image_features) for res in n_intermediate_features
    ])

  def forward(self, features, images):
    outputs = []
    n = len(self.stages)
    current_size = features.shape[-1]
    for i in range(n):
      res = self.stages[i](features, images)
      outputs.append(res)
      if i < n - 1:
        current_size *= 2
        features = torch.nn.functional.interpolate(res["deformed"], current_size, mode="nearest")
      del res
    return outputs