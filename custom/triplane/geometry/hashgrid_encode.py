import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import einops 
import numpy as np 

class HashGridEncodeTorch(nn.Module):
    def __init__(self, 
                 input_dim=2,
                 log_2_hashmap_size=19,
                 num_levels=16,
                 num_feature_per_level=2,
                 start_resolution=16,
                 per_level_scale=1.447269237440378,
                 *args,
                 **kwargs
                 ):
        super().__init__() 
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.b = per_level_scale
        self.start_resolution = start_resolution
        end_resolution = self.start_resolution * (self.b) ** (self.num_levels - 1)
        self.n_output_dims = num_levels * num_feature_per_level 
        self.num_vector_per_level = 2**log_2_hashmap_size
        self.hash_grid = nn.ModuleList()
        for _ in range(num_levels):
            embed = nn.Embedding(self.num_vector_per_level, num_feature_per_level)
            # nn.init.uniform_(embed.weight, a=-1e-4, b=1e-4)
            nn.init.uniform_(embed.weight, a=0.1, b=0.5)
            self.hash_grid.append(embed)
        print("Initialized model hashgrid: ", self.hash_grid)
        self.pi = [1, 2654435761, 805459861]
        if input_dim == 3:
            box_offsets = torch.tensor([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=torch.int32)
            self.register_buffer("box_offsets", box_offsets.reshape(1, 8, 3))
        elif input_dim == 2:
            box_offsets = torch.tensor([[i, j] for i in range(2) for j in range(2)], dtype=torch.int32).reshape(1, 4, 2)
            self.register_buffer("box_offsets", box_offsets)

    def _spatial_hash(self, x):
        assert x.shape[-1] == self.input_dim 
        hash = []
        for i in range(self.input_dim):
            hash.append(x[..., i] * self.pi[i])
        hash = torch.bitwise_xor(*hash)
        hash = torch.remainder(hash, int(self.num_vector_per_level))
        return hash 
    
    def _trilerp(self, x, round_down, resolution, voxel_embeds):
        """
        x: B, 3
        voxel_embeds: B, 8, F -> B, F
        """
        d = ((x * resolution) - round_down).unsqueeze(1) # B, 3
        c00 = voxel_embeds[:, 0] * (1 - d[..., 0]) + voxel_embeds[..., 4] * d[..., 0] # B, F
        c01 = voxel_embeds[:, 1] * (1 - d[..., 0]) + voxel_embeds[..., 5] * d[..., 0]
        c10 = voxel_embeds[:, 2] * (1 - d[..., 0]) + voxel_embeds[..., 6] * d[..., 0]
        c11 = voxel_embeds[:, 3] * (1 - d[..., 0]) + voxel_embeds[..., 7] * d[..., 0]

        c0 = c00 * (1 - d[..., 1]) + c10 * d[..., 1]
        c1 = c01 * (1 - d[..., 1]) + c11 * d[..., 1]
        c = c0 * (1 - d[..., 2]) + c1 * d[..., 2]
        return c
    
    def _bilerp(self, x, round_down, resolution, voxel_embeds):
        d = ((x * resolution) - round_down).unsqueeze(1) # B, 2
        # fix logic bilinear c0 = c00 + c10, c1 = c01 + c11
        # c0 = voxel_embeds[:, 0] * (1 - d[..., 0]) + voxel_embeds[:, 1] * d[..., 0] 
        # c1 = voxel_embeds[:, 2] * (1 - d[..., 0]) + voxel_embeds[:, 3] * d[..., 0]
        c0 = voxel_embeds[:, 0] * (1 - d[..., 0]) + voxel_embeds[:, 2] * d[..., 0] 
        c1 = voxel_embeds[:, 1] * (1 - d[..., 0]) + voxel_embeds[:, 3] * d[..., 0]
        c = c0 * (1 - d[..., 1]) + c1 * d[..., 1]
        return c
        

    def _get_voxel_indices(self, x, current_resolution):
        # round_down, round_up: (B, 3)
        # output: (B, 3, 2**input_dim) and voxel_weight (B, 3, 2**input_dim)
        round_down = torch.floor(x * current_resolution) # (B,3)
        # round_up = (round_down + 1).clamp(min=0, max=current_resolution - 1)
        round_up = round_down + 1 # we don't need to clamp here to avoid edge case in trilerp
        voxel_indices = round_down.unsqueeze(1) + self.box_offsets # B, 8, 3
        voxel_indices = self._spatial_hash(voxel_indices.int()) # B, 8
        return round_down, round_up, voxel_indices
    
    def forward(self, x):
        """
        pipeline: for each level
        1. compute the 2**input_dim point around
            compute round-up and round-down point
            compute 2**input_dim point around -> by iterate
        2. compute the hash for 2**input_dim point and extract vector
        3. interpolate to get the feature vector
        4. concatenate the feature vector
        """
        x_embeds = []
        for i in range(self.num_levels):
            current_resolution = int(self.b**(i) * self.start_resolution)
            round_down, round_up, voxel_indices = self._get_voxel_indices(x, current_resolution)
            voxel_embeds = self.hash_grid[i](voxel_indices) # (B, 8, F)
            if self.input_dim == 3:
                x_embed = self._trilerp(x, round_down, current_resolution, voxel_embeds)
            elif self.input_dim == 2:
                x_embed = self._bilerp(x, round_down, current_resolution, voxel_embeds)
            x_embeds.append(x_embed)
        return torch.cat(x_embeds, dim=-1)

            
