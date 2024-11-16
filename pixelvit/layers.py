
import torch
import torch.nn as nn


Activation = nn.GELU
NormLayer = nn.BatchNorm2d

class DepthwiseConvolution(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, padding_mode="replicate", **kwargs):
        super(DepthwiseConvolution, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, groups=nin, **kwargs)
        self.pointwise = nn.Conv2d(nin, nout,kernel_size=1, padding=0,  **kwargs )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ResidualBlock(nn.Module):
  def __init__(self, out_channels, kernel_size=3, padding=1, depthwise=True, dtype=torch.float32, 
                shrinkage=1.0):
    super().__init__()
    Layer = DepthwiseConvolution if depthwise else nn.Conv2d
    self.layers = nn.Sequential(
        Layer(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dtype=dtype, bias=False),
        NormLayer(out_channels, dtype=dtype),
        Activation(),
        Layer(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dtype=dtype, bias=False),
        NormLayer(out_channels, dtype=dtype),
        Activation(),
    )
    self.shrinkage = shrinkage

  def forward(self, x):
    l = self.layers(x)
    return x + self.shrinkage * l

class ProjectionLayer(nn.Module):
  def __init__(self, n_features_in, n_features_out):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(n_features_in, n_features_out, 1, bias=False),
        NormLayer(n_features_out),
    )

  def forward(self, x):
    return self.layers(x)

class FCLayer(nn.Module):
  def __init__(self,  n_features_in, n_features_out=None, residual=False, bn=True, shrinkage=1.0):
    super().__init__()
    if n_features_out is None:
      n_features_out = n_features_in
    self.residual = residual if n_features_in == n_features_out else False
    self.layers = nn.Sequential(
        nn.Conv2d(n_features_in, n_features_out, 1, bias=not bn),
        Activation(),
    )
    self.shrinkage = shrinkage
    if bn:
      self.bn = NormLayer(n_features_out)
    else:
      self.bn = None

  def forward(self, f):
    res = self.layers(f)
    if self.residual:
      return f + self.shrinkage * res
    if self.bn is None:
      return res
    return self.bn(res)

class ConvolutionalStage(nn.Module):
  def __init__(self, n_features):
    super().__init__()
    self.layers = nn.Sequential(
        ProjectionLayer(n_features * 2, n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
        ResidualBlock(n_features),
    )

  def forward(self, features, image):
    return self.layers(torch.cat([features, image], dim=1))
  