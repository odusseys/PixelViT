from torch import nn
from pixelvit.layers import FCLayer
import kornia
import torch
from pixelvit.model import PixelViT
import bitsandbytes
import json
from pixelvit.preprocessing import make_positional_encodings
import wandb
from PIL import Image
import numpy as np
from pixelvit.util import get_segmentation_map, debug_features
from IPython.display import display

N_SEGMENTATION_CLASSES = 133


class TaskHeads(nn.Module):
  def __init__(self, n_features_in):
    super().__init__()
    def make_head(dim):
      return nn.Sequential(
        nn.Conv2d(n_features_in, n_features_in // 2, 1),
        nn.GELU(),
        nn.Conv2d(n_features_in // 2, n_features_in // 4, 1),
        nn.GELU(),
        nn.Conv2d(n_features_in // 4, n_features_in // 8, 1),
        nn.GELU(),
        nn.Conv2d(n_features_in // 8, dim, 1),
    )

    self.reproduction_head = make_head(3)

    self.depth_head = make_head(1)

    self.segmentation_head = nn.Sequential(
        FCLayer(n_features_in, n_features_in),
        FCLayer(n_features_in, n_features_in),
        nn.Conv2d(n_features_in, N_SEGMENTATION_CLASSES, 1),
    )

    self.diffuse_illumination_head = make_head(3)
    self.diffuse_reflectance_head = make_head(3)
    self.residual_head = make_head(3)
    self.normal_bump_cam_head = make_head(3)

  def forward(self, x):
    return dict(
                depth=self.depth_head(x),
                segmentation=self.segmentation_head(x),
                reproduced=self.reproduction_head(x),
                diffuse_illuminations=self.diffuse_illumination_head(x),
                diffuse_reflectances=self.diffuse_reflectance_head(x),
                residuals=self.residual_head(x),
                normal_bump_cams=self.normal_bump_cam_head(x),
              )




@torch.compile
def structural_loss(x, y):
  return ((x - y) ** 2).mean() + kornia.losses.ssim_loss(x, y, 11)

@torch.compile(dynamic=True)
def gradient_loss(reproduced, truth):
  B, C, W, H = reproduced.shape
  gr = kornia.filters.spatial_gradient(reproduced).reshape((B, 2 * C, W, H))
  gt = kornia.filters.spatial_gradient(truth).reshape((B, 2 * C, W, H))
  return structural_loss(gr, gt)

@torch.compile(dynamic=True)
def laplacian_loss(reproduced, truth):
  lr = kornia.filters.laplacian(reproduced, 3)
  lt = kornia.filters.laplacian(truth, 3)
  return structural_loss(lr, lt)

def reproduction_loss(reproduced, truth, resolution, order=0):
  reproduced = torch.nn.functional.interpolate(reproduced, resolution, mode="bilinear", align_corners=False, antialias=True)
  truth = torch.nn.functional.interpolate(truth, resolution, mode="bilinear", align_corners=False, antialias=True)
  loss = structural_loss(reproduced, truth)
  if order > 0:
    loss += gradient_loss(reproduced, truth)
  if order > 1:
    loss += laplacian_loss(reproduced, truth)
  return loss


def segmentation_loss(logits, ground_truth_probabilities):
  sd = logits.shape[-1] / 128
  window = max(3, int(sd * 2 + 1))

  logits = kornia.filters.gaussian_blur2d(logits, window, (sd,sd))
  batch_size, n_classes, height, width = logits.shape
  ground_truth_probabilities = nn.functional.interpolate(ground_truth_probabilities, (height, width), mode="bilinear", align_corners=False, antialias=True)
  logits = logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
  ground_truth_probabilities = ground_truth_probabilities.permute(0, 2, 3, 1).reshape(batch_size * height * width, n_classes)

  return nn.functional.cross_entropy(logits, ground_truth_probabilities).mean()


def init_wandb_run(run_type, slug):
  run = wandb.init(
    project="feature-upscaling",
    config={
        "run_type": run_type,
        "slug": slug,
    }
  )
  return run


def full_loss(results_list, batch):
  rpr = 0
  dptl = 0
  segl = 0
  diffl = 0
  diffr = 0
  resl = 0
  norml = 0
  gridl = 0
  n = 0

  for results in results_list:
    gridl += 0.0 # 0.1 * grid_loss(results["field"])
    resolution = batch["images"].shape[-1]
    while resolution > 16:
      r = results["deformed_head_results"]
      n += 1

      rpr += reproduction_loss(r["reproduced"], batch["images"], resolution, order=0)
      dptl += 2 * reproduction_loss(r["depth"], batch["depth"], resolution, order=2)
      segl += 0.2 * segmentation_loss(r["segmentation"], batch["segmentation"])

      if batch["diffuse_illuminations"] is not None:
        diffl += reproduction_loss(r["diffuse_illuminations"], batch["diffuse_illuminations"], resolution, order=2)
        diffr += reproduction_loss(r["diffuse_reflectances"], batch["diffuse_reflectances"], resolution, order=2)
        resl += reproduction_loss(r["residuals"], batch["residuals"], resolution, order=2)
        norml += reproduction_loss(r["normal_bump_cams"], batch["normal_bump_cams"], resolution, order=2)

      del r
      resolution //= 2

    del results

  gridl /= len(results_list)
  rpr = rpr / n
  dptl = dptl / n
  segl = segl / n
  diffl = diffl / n
  diffr = diffr / n
  resl = resl / n
  norml = norml / n


  res = dict(
      reproduction=rpr,
      depth_loss=dptl,
      segmentation_loss=segl,
      diffl=diffl,
      diffr=diffr,
      resl=resl,
      norml=norml,
      gridl=gridl,
      loss=(rpr + dptl + segl + diffl + diffr + resl + norml + gridl) / 7,
  )
  del rpr
  del dptl
  del segl
  del diffl
  del diffr
  del resl
  del norml
  del gridl
  return res


class LossTracker:
  def __init__(self, names, decay=0.99):
    self.names = names
    self.losses = None
    self.decay = decay

  def track(self, losses):
    if self.losses is None:
      self.losses = [x if type(x) == float else x.detach() for x in losses]
    else:
      for i in range(len(losses)):
        l = losses[i]
        loss = l if type(l) == float else l.detach()
        self.losses[i] = self.losses[i] * self.decay + loss * (1 - self.decay)
        del l

  def debug(self):
    for i in range(len(self.names)):
      print(self.names[i], ": ", self.losses[i])



def debug_step(batch, results, running_loss, n, display_size, loss_tracker, use_wandb):
  with torch.no_grad():
    def make_reproduced(x):
      x = (torch.clamp(x[0], 0, 1).permute((1, 2, 0)) * 255.0).detach().cpu().numpy().astype(np.uint8)
      return Image.fromarray(x).resize((display_size, display_size))

    reproduced = [batch["originals"][0].resize((display_size, display_size)),
                  *[make_reproduced(r["deformed_head_results"]["reproduced"]) for r in results]]

    reproduced = Image.fromarray(np.hstack(reproduced).astype(np.uint8))

    depth_reproduced = [batch["depth"][0],
                        *[torch.clamp(r["deformed_head_results"]["depth"][0], 0, 1) for r in results]]
    depth_reproduced = [Image.fromarray((x.squeeze() * 255.0).detach().cpu().numpy().astype(np.uint8)).resize((display_size, display_size)) for x in depth_reproduced]
    depth_reproduced = Image.fromarray(np.hstack(depth_reproduced).astype(np.uint8)).convert("RGB")

    def make_seg(x):
      x = torch.nn.functional.interpolate(x.unsqueeze(0), size=batch["segmentation"].shape[-1], mode="nearest").squeeze(0)
      x = torch.argmax(x, dim=0).to(dtype=torch.uint8).squeeze()
      return x

    segmentation_truth = torch.argmax(batch["segmentation"][0], dim=0).to(dtype=torch.uint8).squeeze()
    segmentation_reproduced = [segmentation_truth, *[make_seg(r["deformed_head_results"]["segmentation"][0]) for r in results]]
    del segmentation_truth
    segmentation_reproduced = [get_segmentation_map(batch["images"][0], x).resize((display_size,display_size)) for x in segmentation_reproduced]
    segmentation_reproduced = Image.fromarray(np.hstack(segmentation_reproduced).astype(np.uint8))

    feature_debug = debug_features(batch["features"][0], [r["deformed"][0] for r in results], display_size)

    debug_image = Image.fromarray(np.vstack([feature_debug, reproduced, depth_reproduced, segmentation_reproduced]).astype(np.uint8))

    if use_wandb:
      wandb.log({
          "loss": running_loss,
          "debug_image": wandb.Image(debug_image),
      })
    print("Iteration", n, ":", running_loss, )
    loss_tracker.debug()
    display(debug_image)

def train_onestep(config,
                  lr = 1e-2,
                  batch_size = 4,
                  accumulation=1,
                  max_iterations=None,
                  debug_interval=51,
                  use_wandb=False,
                  display_size=128,
                  annealing_restart_iterations=1000,
                  annealing_min_learning_rate=1e-6,
                  checkpoint=None):
  save_interval = annealing_restart_iterations * accumulation
  start_size = config["start_size"]
  target_size = config["target_size"]
  # up_factor = int(math.log2(target_size // start_size)) + 1

  upscaler = PixelViT(config).to("cuda")


  # upscaler = torch.compile(upscaler, dynamic=True)
  task_heads = TaskHeads(config["n_features_in"]).to("cuda")

  if checkpoint is not None:
    upscaler.load_state_dict(torch.load(f"upscaler-{checkpoint}"))
    task_heads.load_state_dict(torch.load(f"task_heads-{checkpoint}"))
    task_heads.load_state_dict(torch.load(f"optimizer-{checkpoint}"))

  upscaler.train()
  task_heads.train()

  if use_wandb:
    run = init_wandb_run("feature_fixer", "reproduction")
    wandb.watch(upscaler, log_freq=100)

  if use_wandb:
    with open("config.json", "w+") as f:
      json.dump(config, f)
      wandb.save("config.json")

  # OPTIMIZER CONFIG
  optimizer = bitsandbytes.optim.AdamW8bit(
[      {"params": upscaler.parameters(), "lr": lr},
      {"params": task_heads.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.0)
  scaler = torch.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=annealing_restart_iterations, eta_min=annealing_min_learning_rate)
  loss_tracker = LossTracker(["reprod", "depth", "segmentation", "diffl", "diffr", "resl", "norml", "gridl",
                               "loss"], decay=0.98)
  running_loss = None
  pe = make_positional_encodings(batch_size, target_size, config["n_frequencies"])
  n = 0
  try:
    for epoch in range(5):
        for batch in get_mixed_data(batch_size, start_size, target_size, config["n_wavelets"], blur=None):
            if n == max_iterations:
              return
            n += 1
            with torch.autocast(device_type='cuda', dtype=torch.float16), torch.set_grad_enabled(True):
              image_features = torch.cat([batch["wavelets"], pe], dim=1)

              results = upscaler(batch["features"], image_features)
              for r in results:
                r["deformed_head_results"] = task_heads(r["deformed"])
              del image_features

              losses = full_loss(results, batch)
              loss = losses["loss"] / accumulation

              scaler.scale(loss).backward()
              if n % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler.step()

              if n % debug_interval == 0:
                debug_step(batch, results, running_loss, n, display_size, loss_tracker, use_wandb)
                print("SCHEDULER LR", scheduler.get_lr())

              loss_tracker.track([
                  losses["reproduction"],
                  losses["depth_loss"],
                  losses["segmentation_loss"],
                  losses["diffl"],
                  losses["diffr"],
                  losses["resl"],
                  losses["norml"],
                  losses["gridl"],
                  losses["loss"]])

              if running_loss is None:
                running_loss = accumulation * loss.detach()
              else:
                running_loss = 0.99 * running_loss + 0.01 *  accumulation * loss.detach()

            if n % save_interval == 0:
              torch.save(upscaler.state_dict(), f"upscaler-{epoch}-{n}.pth")
              torch.save(task_heads.state_dict(), f"task_heads-{epoch}-{n}.pth")
              torch.save(optimizer.state_dict(), f"optimizer-{epoch}-{n}.pth")
              if use_wandb:
                wandb.save(f"upscaler-{epoch}-{n}.pth")
                wandb.save(f"task_heads-{epoch}-{n}.pth")
                wandb.save(f"optimizer-{epoch}-{n}.pth")


            del batch
            del loss
            del results
            del losses


          # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

  except:

    if use_wandb:
      run.finish()
    del pe
    del batch
    del loss
    del optimizer
    del upscaler
    del scaler
    del results
    del loss_tracker
    del losses

    raise