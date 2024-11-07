import numpy as np
import torch

locations_cache = {}

def make_base_locations(batch_size, size):
  with torch.no_grad():
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
