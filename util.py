
import torch
import numpy as np


# https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates
def project_sphere(xyz):
    rho = torch.norm(xyz, dim=1, keepdim=False)
    theta = torch.atan2(xyz[..., 1], xyz[..., 0])
    xy_norm = torch.norm(xyz[:, 0:2], dim=1, keepdim=False)
    phi = torch.atan2(-xyz[..., 2], xy_norm)
    return torch.stack([theta, phi, rho], dim=-1)


def visualize(semantic):
    color_map = np.array([[255, 255, 255], [0, 0, 255], [128, 0, 0], [255, 0, 255], [0, 128, 0],
                          [255, 0, 0], [128, 0, 128], [0, 0, 128], [128, 128, 0]], dtype=np.uint8)
    color_map = torch.from_numpy(color_map)
    semantic = semantic.squeeze(dim=-1)
    shape = list(semantic.shape)
    shape.append(3)
    image = torch.zeros(shape, dtype=torch.float32)
    for i in range(1, len(color_map)):
        image[semantic == i] = color_map[i] / 255.0
    return image
