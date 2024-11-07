import torch
import math

dino_model = 's'

features_sizes = {
    's': 384,
    'b': 768,
    'l': 1024,
    'g': 1536
}

N_FEATURES = features_sizes[dino_model]


# monkeypatch utility for striding
def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        new_w = int(math.sqrt(x.shape[1])) * self.patch_size
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, new_w, new_w)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

class DinoV2:
  def __init__(self, stride=None):
    dino_backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{dino_model}14_reg').to("cuda", torch.float32)
    dino_backbone.eval()
    if stride is not None:
      dino_backbone.patch_embed.proj.stride = (stride, stride)
      dino_backbone.prepare_tokens_with_masks = types.MethodType(prepare_tokens_with_masks, dino_backbone)
    dino_backbone.requires_grad_(False)
    self.dino_backbone = torch.compile(dino_backbone)
    pass

  def prepare_images(self, images, image_check=True):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device="cuda") / 255.0
    return tensors

  def get_features_for_tensor(self, images, image_check=True):
    with torch.no_grad():
      res = self.dino_backbone.get_intermediate_layers(images, n=1)[-1]
      grid_size = int(math.sqrt(res.shape[1]))
      res = res.reshape((res.shape[0], grid_size, grid_size, N_FEATURES)).permute((0, 3, 1, 2))
      del images
      return res

  def get_features(self, images, image_check=True, one_by_one=True):
    images = self.prepare_images(images, image_check)
    return self.get_features_for_tensor(images, image_check)
