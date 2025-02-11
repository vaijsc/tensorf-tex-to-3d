import math
import itertools

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

class MultiTriMipEncoding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config,
        include_xyz: bool = False,
    ):
        super().__init__()
        self.config = config
        self.n_input_dims = in_channels
        self.n_levels = config.n_levels
        self.plane_size = config.plane_size
        self.feature_dim = config.feature_dim
        self.multiscale_res_multipliers = config.multiscale_res or [1]
        if config.concat_features_across_scales:
            self.n_output_dims = 3 * config.feature_dim * len(config.multiscale_res)
        else:
            self.n_output_dims = 3 * config.feature_dim
        self.concat_features = config.concat_features_across_scales
        self.include_xyz = include_xyz

        self.grids = nn.ModuleList()

        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.config.copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = config["plane_size"] * res
                
            gp = self._init_grid_param(
                grid_nd=2,
                in_dim=3,
                out_dim=config["feature_dim"],
                reso=config["resolution"],
            ) # 3, H, W, D

            self.grids.append(gp)
        print(f"Initialized model grids: {self.grids}")
        self.log2_plane_size_lowest = math.log2(self.plane_size)
        self.register_buffer("log_2_plane_size", 
            torch.tensor([self.log2_plane_size_lowest + i for i in range(len(config.multiscale_res))]))
        

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def _init_grid_param(self,
                        grid_nd: int,
                        in_dim: int,
                        out_dim: int,
                        reso,
                        a: float = -1e-2,
                        b: float = 1e-2,
                        n_components: int = 1):
            grid_coefs = nn.ParameterList()
            new_grid_coef = nn.Parameter(torch.empty(
                    [n_components * in_dim, reso, reso, out_dim]
                ))
            nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)
            return grid_coefs

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
        multi_scale_interp = [] if self.concat_features else 0.
        for scale_id, grid in enumerate(self.grids):
            enc = nvdiffrast.torch.texture(
                grid[0],
                decomposed_x,
                mip_level_bias=level+self.log_2_plane_size[scale_id].item(),
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
            if self.concat_features:
                multi_scale_interp.append(enc)
            else:
                multi_scale_interp += enc 
        if self.concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

class MipTensorVMSplit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config,
        include_xyz: bool = False,
    ):
        super().__init__()
        self.config = config
        self.density_n_comp = self.config.get("density_n_comp", 8)
        self.app_n_comp = self.config.get("appearance_n_comp", 24)
        self.app_dim = self.config.get("app_dim", 27)
        self.density_shift = self.config.get("density_shift", -10)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.gridSize = config["resolution"]
        
        self.init_svd_volume(self.gridSize[0])
        self.n_output_dims = self.app_dim
        self.n_input_dims = in_channels
        self.log2_plane_size = math.log2(self.gridSize[0])
        

    def init_svd_volume(self, res):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)

    def _init_grid_param(self,
                        grid_nd: int,
                        in_dim: int,
                        out_dim: int,
                        reso,
                        a: float = -1e-2,
                        b: float = 1e-2,
                        n_components: int = 1):
            grid_coefs = nn.ParameterList()
            new_grid_coef = nn.Parameter(torch.empty(
                    [n_components * in_dim, reso, reso, out_dim]
                ))
            nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)
            return grid_coefs

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
        multi_scale_interp = [] if self.concat_features else 0.
        for scale_id, grid in enumerate(self.grids):
            enc = nvdiffrast.torch.texture(
                grid[0],
                decomposed_x,
                mip_level_bias=level+self.log_2_plane_size[scale_id].item(),
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
            if self.concat_features:
                multi_scale_interp.append(enc)
            else:
                multi_scale_interp += enc 
        if self.concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp
    

def get_multitrimip(n_input_dims: int, config) -> nn.Module:
    encoding = MultiTriMipEncoding(n_input_dims, config)
    return encoding

def get_trimip(n_input_dims: int, config) -> nn.Module:
    encoding = TriMipEncoding(n_input_dims, config)
    return encoding


