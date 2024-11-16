import torch
import math
import torchvision


class DinoV2:
  def __init__(self, dino_model):
    dino_backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{dino_model}14_reg').to("cuda", torch.float32)
    dino_backbone.eval()
    dino_backbone.requires_grad_(False)
    self.dino_backbone = torch.compile(dino_backbone)
    pass

  def prepare_images(self, images):
    tensors = [torchvision.transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device="cuda") / 255.0
    return tensors

  def get_features_for_tensor(self, images):
    with torch.no_grad():
      res = self.dino_backbone.get_intermediate_layers(images, n=1)[-1]
      grid_size = int(math.sqrt(res.shape[1]))
      n_features = res.shape[-1]
      res = res.reshape((res.shape[0], grid_size, grid_size, n_features)).permute((0, 3, 1, 2))
      del images
      return res

  def get_features(self, images):
    images = self.prepare_images(images)
    return self.get_features_for_tensor(images)

  