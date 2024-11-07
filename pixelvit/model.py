import torch
import torch.nn as nn
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

    nn.init.normal_(last_layer.weight, mean=0.0, std=0.01, generator=None)
    nn.init.zeros_(last_layer.bias)



  def forward(self, features, image):
    image = self.image_encoder(image)
    image = torch.nn.functional.interpolate(image, features.shape[-1], mode="bilinear")
    f = self.feature_encoder(features)
    f = transformer(f, image)
    base_locations = make_base_locations(image.shape[0], image.shape[-1]).permute((0, 3, 1, 2))
    field = base_locations + self.deformer(f)
    field = field.permute((0, 2, 3, 1))
    return torch.nn.functional.grid_sample(features, field, padding_mode="border", mode="bilinear", align_corners=False)



DEFAULT_PE_FREQUENCIES = 64
DEFAULT_N_WAVELETS = 8
DEFAULT_IMAGE_FEATURES = 6 + (3 * 3 * DEFAULT_N_WAVELETS) + 2 * DEFAULT_PE_FREQUENCIES

class PixelViT(nn.Module):
  def __init__(self, 
                n_features_in=N_FEATURES,
                n_image_features=DEFAULT_IMAGE_FEATURES, 
                n_intermediate_features=256,
                start_size=32, 
                target_size=256):
    super().__init__()

    n_resizes = int(math.log2(target_size // start_size)) + 1

    self.stages = nn.ModuleList([
        PixelViTStage(n_intermediate_features, n_features_in, n_image_features) for i in range(n_resizes)
    ])

  def forward(self, features, images):
    outputs = []
    n = len(self.stages)
    current_size = features.shape[-1]
    for i in range(n):
      features = self.stages[i](features, images)
      outputs.append(features)
      if i < n - 1:
        current_size *= 2
        features = torch.nn.functional.interpolate(features, current_size, mode="nearest")
    return outputs
