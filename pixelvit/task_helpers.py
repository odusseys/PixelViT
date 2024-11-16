from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation
import torch


depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()
depth_anything = torch.compile(depth_anything)

depth_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ], device="cuda").reshape((1, 3, 1, 1))

depth_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ], device="cuda").reshape((1, 3, 1, 1))

depth_size = 518

def preprocess_image_for_depth(image_tensor):
  image_tensor = (image_tensor - depth_image_mean) / depth_image_std
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(depth_size, depth_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  return image_tensor


def get_depth(images):
  size = images.shape[-1]
  with torch.no_grad():
    inputs = preprocess_image_for_depth(images)
    outputs = depth_anything(pixel_values=inputs)
    predicted_depth = outputs.predicted_depth
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )
    predicted_depth[torch.isnan(predicted_depth)] = 0
    m = predicted_depth.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    M = predicted_depth.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    predicted_depth = (predicted_depth - m) / (M - m + 1e-5)
    return predicted_depth
  

# load Mask2Former fine-tuned on COCO panoptic segmentation
segmentation_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-panoptic").cuda()

def make_segmentation_probabilities(outputs):
  class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
  masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

  # Scale back to preprocessed image size - (384, 384) for all models
  masks_queries_logits = torch.nn.functional.interpolate(
      masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
  )

  # Remove the null class `[..., :-1]`
  masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
  masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

  # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
  segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
  return segmentation

segmentation_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ], device="cuda").reshape((1, 3, 1, 1))

segmentation_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ], device="cuda").reshape((1, 3, 1, 1))

segmentation_image_size = 384

def preprocess_image_for_segmentation(image_tensor):
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(segmentation_image_size, segmentation_image_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  image_tensor = (image_tensor - segmentation_image_mean) / segmentation_image_std

  return image_tensor

def get_segmentation(images):
  size = images.shape[-1]
  with torch.no_grad():
    inputs = preprocess_image_for_segmentation(images)
    outputs = segmentation_model(pixel_values=inputs)
    probas = make_segmentation_probabilities(outputs)
    return torch.nn.functional.interpolate(
        probas,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )
