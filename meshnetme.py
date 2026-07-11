import torch
import torch.nn as nn
import torch.nn.functional as F
from meshnet_gn import set_channel_num, construct_layer, init_weights
from torch.utils.checkpoint import checkpoint_sequential
import json


class MeshNetME(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        num_experts=None,
        average_outputs=True,
        train_experts=None,
        eval_experts=None,
    ):
        super().__init__()
        if isinstance(config_file, str):
            config_files = [config_file] * (num_experts or 1)
        else:
            config_files = list(config_file)
            if num_experts is not None and num_experts != len(config_files):
                raise ValueError("num_experts must match len(config_file) when config_file is a list")

        self.num_experts = len(config_files)
        self.average_outputs = average_outputs
        self.train_experts = train_experts or self.num_experts
        self.eval_experts = eval_experts or self.num_experts
        if self.num_experts < 1 or self.train_experts < 1 or self.eval_experts < 1:
            raise ValueError("ME requires at least one expert")

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
            layers[-1] = layers[-1][0]
            tower = nn.Sequential(*layers)
            init_weights(tower)
            towers[f"tower_{i}"] = tower
        self.towers = nn.ModuleDict(towers)

    def selected_tower_names(self):
        names = list(self.towers.keys())
        if self.training:
            count = min(self.train_experts, self.num_experts)
            if count == self.num_experts:
                return names
            order = torch.randperm(self.num_experts).tolist()
            return [names[i] for i in order[:count]]
        return names[: min(self.eval_experts, self.num_experts)]

    def weight_diversity_loss(self):
        """Penalize experts using similar convolution kernels.

        This is intentionally weight-only: it adds no activation memory and is
        cheap compared with the 3D forwards. The loss is near zero when matching
        expert kernels are orthogonal in weight space.
        """
        tower_convs = [
            [module for module in tower.modules() if isinstance(module, nn.Conv3d)]
            for tower in self.towers.values()
        ]
        if len(tower_convs) < 2:
            return torch.zeros((), device=next(self.parameters()).device)

        losses = []
        for layer_convs in zip(*tower_convs):
            weights = [conv.weight.flatten() for conv in layer_convs]
            weights = torch.stack([F.normalize(weight, dim=0) for weight in weights])
            similarity = weights @ weights.t()
            mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
            losses.append(similarity[mask].pow(2).mean())
        return torch.stack(losses).mean() if losses else torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x):
        selected_names = self.selected_tower_names()
        tower_sum = self.towers[selected_names[0]](x)
        for name in selected_names[1:]:
            tower_sum = tower_sum + self.towers[name](x)
        if self.average_outputs:
            tower_sum = tower_sum / len(selected_names)
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
        if not getattr(self, "use_checkpoint", True):
            return model(x)
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            model, len(model), y, preserve_rng_state=False, use_reentrant=False
        )
        return y

    def eval_forward(self, x, model):
        """Forward pass"""
        model.eval()
        with torch.inference_mode():
            x = model(x)
        return x

    def forward(self, x):
        selected_names = self.selected_tower_names()
        if self.training:
            tower_sum = self.train_forward(x, self.towers[selected_names[0]])
            for name in selected_names[1:]:
                tower_sum = tower_sum + self.train_forward(x, self.towers[name])
        else:
            with torch.inference_mode():
                tower_sum = self.eval_forward(x, self.towers[selected_names[0]])
                for name in selected_names[1:]:
                    tower_sum = tower_sum + self.eval_forward(x, self.towers[name])

        if self.average_outputs:
            tower_sum = tower_sum / len(selected_names)
        return tower_sum


class MeshNetME_checkpoint(CheckpointMixin, MeshNetME):
    pass
