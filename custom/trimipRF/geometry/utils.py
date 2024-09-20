import math

import torch
from torch import nn
import nvdiffrast.torch

from threestudio.models.networks import CompositeEncoding
from threestudio.utils.typing import *


class TriMipEncoding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config,
        include_xyz: bool = False,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_input_dims = in_channels
        self.n_levels = config.n_levels
        self.plane_size = config.plane_size
        self.feature_dim = config.feature_dim
        self.include_xyz = include_xyz

        self.register_parameter(
            "fm",
            nn.Parameter(torch.zeros(3, self.plane_size, self.plane_size, self.feature_dim)),
        )
        self.init_parameters()
        self.n_output_dims = self.feature_dim * 3
        self.log2_plane_size = math.log2(self.plane_size)

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level):
        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
        decomposed_x = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2
        if 0 == self.n_levels:
            level = None
        else:
            # assert level.shape[0] > 0, [level.shape, x.shape]
            level = torch.stack([level, level, level], dim=0)
            level = torch.broadcast_to(
                level, decomposed_x.shape[:3]
            ).contiguous()
        enc = nvdiffrast.torch.texture(
            self.fm,
            decomposed_x,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )  # 3xNx1xC
        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                self.feature_dim * 3,
            )
        )  # Nx(3C)
        return enc


def get_trimip(n_input_dims: int, config) -> nn.Module:
    encoding = TriMipEncoding(n_input_dims, config)
    return encoding