import torch
import torch.nn as nn
import torch.nn.functional as F
from meshnet_gn import set_channel_num, construct_layer
from torch.utils.checkpoint import checkpoint_sequential
import json


class MeshNetME(nn.Module):
    def __init__(self, in_channels, n_classes, channels, config_file):
        super().__init__()
        self.num_experts = len(config_file)

        # Initialize the towers
        towers = {}
        for i, config in enumerate(config_file):
            with open(config, "r") as f:
                config = set_channel_num(
                    json.load(f), in_channels, n_classes, channels
                )
            layers = [
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config["gelu"],
                    **block_kwargs,
                )
                for block_kwargs in config["layers"]
            ]
            layers[-1] = layers[-1][0]
            towers[f"tower_{i}"] = nn.Sequential(*layers)
        self.towers = nn.ModuleDict(towers)

    def forward(self, x):
        # Compute the outputs of each expert
        # option 1
        # tower_sum = sum([tower(x) for tower in self.towers.values()])
        # memory efficient option 2
        # tower_sum = next(iter(self.towers.values()))(x)
        # # Add the outputs of the remaining towers
        # for tower in list(self.towers.values())[1:]:
        #     tower_sum += tower(x)

        tower_outputs = [tower(x) for tower in self.towers.values()]
        tower_sum = torch.sum(torch.stack(tower_outputs, dim=1), dim=1)

        return tower_sum


class MeshNetME_(nn.Module):
    def __init__(self, in_channels, n_classes, channels, config_files):
        super().__init__()
        self.num_experts = len(config_files)

        # Initialize the towers
        towers = {}
        for i, config_file in enumerate(config_files):
            with open(config_file, "r") as f:
                config = set_channel_num(
                    json.load(f), in_channels, n_classes, channels
                )
            layers = [
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config["gelu"],
                    **block_kwargs,
                )
                for block_kwargs in config["layers"]
            ]
            del layers[-1]
            towers[f"tower_{i}"] = nn.Sequential(*layers)
        self.towers = nn.ModuleDict(towers)
        layers = [
            nn.GroupNorm(
                num_groups=channels,
                num_channels=channels,
            ),
            nn.Conv3d(channels, n_classes, kernel_size=1),
        ]
        self._last_layer = nn.Sequential(*layers)

    def forward(self, x):
        # Compute the outputs of each expert
        # option 1
        # tower_sum = sum([tower(x) for tower in self.towers.values()])
        # memory efficient option 2
        # tower_sum = next(iter(self.towers.values()))(x)
        # # Add the outputs of the remaining towers
        # for tower in list(self.towers.values())[1:]:
        #     tower_sum += tower(x)

        tower_outputs = [tower(x) for tower in self.towers.values()]
        tower_sum = torch.sum(torch.stack(tower_outputs, dim=1), dim=1)
        # Apply the final convolution
        output = self._last_layer(tower_sum)
        return output


class CheckpointMixin:
    def train_forward(self, x, model):
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            model, len(model), y, preserve_rng_state=False
        )
        return y

    def eval_forward(self, x, model):
        """Forward pass"""
        model.eval()
        with torch.inference_mode():
            x = model(x)
        return x

    def forward(self, x):
        if self.training:
            # tower_outputs = [
            #     self.train_forward(x, tower) for tower in self.towers.values()
            # ]
            tower_sum = self.train_forward(x, next(iter(self.towers.values())))
            # Add the outputs of the remaining towers
            for tower in list(self.towers.values())[1:]:
                tower_sum += self.train_forward(x, tower)
        else:
            # tower_outputs = [
            #     self.eval_forward(x, tower) for tower in self.towers.values()
            # ]
            with torch.inference_mode():
                tower_sum = self.eval_forward(
                    x, next(iter(self.towers.values()))
                )
                # Add the outputs of the remaining towers
                for tower in list(self.towers.values())[1:]:
                    tower_sum += self.eval_forward(x, tower)

        # tower_sum = torch.sum(torch.stack(tower_outputs, dim=1), dim=1)

        return tower_sum


class MeshNetME_checkpoint(CheckpointMixin, MeshNetME):
    pass
