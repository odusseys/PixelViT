import torch
import torchvision
import torch.nn as nn
import random
import os
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from pixelvit.task_helpers import get_segmentation, get_depth

imagenet = load_dataset("timm/imagenet-22k-wds", streaming=True)["train"]


def collate_function(x):
  return x[0]

imagenet = DataLoader(imagenet, batch_size=1, collate_fn=collate_function, num_workers=32)

def load_imagenet(batch_size):
  images = []
  blocks = []
  n = 0
  for x in imagenet:
    x = x["jpg"].convert("RGB")
    l = min(x.height, x.width)
    x = x.crop((0, 0, l ,l ))
    x = x.resize((256, 256))
    block = None
    images.append(x)
    blocks.append(block)
    n += 1
    if n == batch_size:
      yield dict(images=images, blocks=blocks)
      images = []
      blocks = []
      n = 0

def list_hypersim_images(path):
  res = []
  for folder in os.listdir(f"{path}"):
    try:
      images_folders = os.listdir(f"{path}/{folder}/images")
      if "final" in images_folders[0]:
        final_folder = images_folders[0]
        geometry_folder = images_folders[1]
      else:
        final_folder = images_folders[1]
        geometry_folder = images_folders[0]
      final_files = os.listdir(f"{path}/{folder}/images/{final_folder}")
      frames = set(f.split(".")[1] for f in final_files)
      for frame in frames:
        res.append((folder, final_folder, geometry_folder, frame))
    except:
      continue
  # shuffle with fixed seed
  random.seed(42)
  random.shuffle(res)
  return res


def load_hypersim_images(path):
  for folder, final_folder, geometry_folder, frame in list_hypersim_images(path):
    try:
      image = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.color.jpg").convert("RGB")
      diffuse_illumination = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.diffuse_illumination.jpg").convert("RGB")
      diffuse_reflectance = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.diffuse_reflectance.jpg").convert("RGB")
      residual = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.residual.jpg").convert("RGB")
      # semantic = Image.open(f"{path}/{folder}/images/{geometry_folder}/frame.{frame}.semantic.png")
      normal_bump_cam = Image.open(f"{path}/{folder}/images/{geometry_folder}/frame.{frame}.normal_bump_cam.png").convert("RGB")
    except:
      continue
    yield image, diffuse_illumination, diffuse_reflectance, residual, normal_bump_cam

def load_hypersim(batch_size, path="hypersim"):
  images = []
  diffuse_illuminations = []
  diffuse_reflectances = []
  residuals = []
  normal_bump_cams = []
  n = 0
  for image, diffuse_illumination, diffuse_reflectance, residual, normal_bump_cam in load_hypersim_images(path):
    images.append(image)
    diffuse_illuminations.append(diffuse_illumination)
    diffuse_reflectances.append(diffuse_reflectance)
    residuals.append(residual)
    normal_bump_cams.append(normal_bump_cam)
    n += 1
    if n == batch_size:
      yield dict(
          images=images,
          diffuse_illuminations=diffuse_illuminations,
          diffuse_reflectances=diffuse_reflectances,
          residuals=residuals,
          normal_bump_cams=normal_bump_cams
      )
      images = []
      diffuse_illuminations = []
      diffuse_reflectances = []
      residuals = []
      normal_bump_cams = []
      n = 0


MINIMAL_IMAGE_SIZE = 224
MINIMAL_GRID_SIZE = MINIMAL_IMAGE_SIZE // 14

def make_batch_data(batch, starting_size, target_size, n_wavelets=7):

    def to_tensor(images):
        return torch.stack([torchvision.transforms.PILToTensor()(img) for img in images]).cuda() / 255.0

    def target_resize(x):
        return nn.functional.interpolate(x, (target_size, target_size), mode="bilinear", align_corners=False, antialias=True)


    originals = [u.resize((512, 512)) for u in batch["images"]]
    images = to_tensor(originals)

    factor = starting_size // MINIMAL_GRID_SIZE

    images_small = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True)
    # add small noise to remove positional encoding artifacts
    images_small = (images_small + torch.randn_like(images_small) * 0.05).clamp(0, 1)

    images = target_resize(images)
    segmentation = get_segmentation(images)
    depth = get_depth(images)

    if "diffuse_illuminations" in batch:
        diffuse_illuminations = target_resize(to_tensor(batch["diffuse_illuminations"]))
        diffuse_reflectances = target_resize(to_tensor(batch["diffuse_reflectances"]))
        residuals = target_resize(to_tensor(batch["residuals"]))
        normal_bump_cams = target_resize(to_tensor(batch["normal_bump_cams"]))
    else:
        diffuse_illuminations = None
        diffuse_reflectances = None
        residuals = None
        normal_bump_cams = None


    res = dict(images=images,
               depth=depth,
               segmentation=segmentation,
               diffuse_illuminations=diffuse_illuminations,
               diffuse_reflectances=diffuse_reflectances,
               residuals=residuals,
               normal_bump_cams=normal_bump_cams,
               originals=originals)
    del images_small
    del images
    del depth
    del segmentation
    del diffuse_illuminations
    del diffuse_reflectances
    del residuals
    del normal_bump_cams

    return res

def get_mixed_data(batch_size, starting_size, target_size, n_wavelets=7, hypersim=True, imagenet=True):
  hypersim_iterator = load_hypersim(batch_size)
  imagenet_iterator = load_imagenet(batch_size)

  n = 0
  while True:
    n += 1
    if n % 2 == 0:
      if not hypersim:
        continue
      x = next(hypersim_iterator, None)
      if x is None:
        hypersim_iterator = load_hypersim(batch_size)
        x = next(hypersim_iterator, None)
    else:
      if not imagenet:
        continue
      x = next(imagenet_iterator, None)
      if x is None:
        imagenet_iterator = load_imagenet(batch_size)
        x = next(imagenet_iterator, None)

    yield make_batch_data(x, starting_size, target_size, n_wavelets)
