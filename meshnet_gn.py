from collections import OrderedDict
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.utils.checkpoint import checkpoint_sequential
import json
import copy


class MixedConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MixedConv3d, self).__init__()

        nondilated_channels = kwargs.pop("nondilated_channels")
        out_channels = kwargs.pop("out_channels")

        self.conv2 = nn.Conv3d(
            *args,
            **kwargs,
            out_channels=out_channels - nondilated_channels,
        )
        padding = kwargs.pop("padding")
        dilation = kwargs.pop("dilation")
        self.conv1 = nn.Conv3d(
            *args,
            **kwargs,
            out_channels=nondilated_channels,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat((x1, x2), dim=1)


class FusedConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FusedConv3d, self).__init__()

        self.conv2 = nn.Conv3d(*args, **kwargs)
        padding = kwargs.pop("padding")
        dilation = kwargs.pop("dilation")
        self.conv1 = nn.Conv3d(
            *args,
            **kwargs,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2


def set_channel_num(config, in_channels, n_classes, channels):
    """
    Takes a configuration json for a convolutional neural network of MeshNet architecture and changes it to have the specified number of input channels, output classes, and number of channels that each layer except the input and output layers have.

    Args:
        config (dict): The configuration json for the network.
        in_channels (int): The number of input channels.
        n_classes (int): The number of output classes.
        channels (int): The number of channels that each layer except the input and output layers will have.

    Returns:
        dict: The updated configuration json.
    """
    # input layer
    config["layers"][0]["in_channels"] = in_channels
    config["layers"][0]["out_channels"] = channels

    # output layer
    config["layers"][-1]["in_channels"] = channels
    config["layers"][-1]["out_channels"] = n_classes

    # hidden layers
    for layer in config["layers"][1:-1]:
        layer["in_channels"] = layer["out_channels"] = channels

    return config


def construct_layer(dropout_p=0, bnorm=True, gelu=False, affine=False, *args, **kwargs):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        # layers.append(
        #     nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        # )
        layers.append(
            nn.GroupNorm(
                num_groups=kwargs["out_channels"],
                num_channels=kwargs["out_channels"],
                affine=affine,
            )
        )

    layers.append(nn.GELU() if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def construct_mixedlayer(
    dropout_p=0, bnorm=True, gelu=False, highreslayers=1, *args, **kwargs
):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    if kwargs["dilation"] > 1:
        layers.append(
            MixedConv3d(*args, **kwargs, nondilated_channels=highreslayers)
        )
    else:
        layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def construct_fusedlayer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    if kwargs["dilation"] > 1:
        layers.append(FusedConv3d(*args, **kwargs))
    else:
        layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def init_weights(model, relu=True):
    """Set weights to be xavier normal for all Convs"""
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            if relu:
                #nn.init.xavier_normal_(
                #    m.weight, gain=nn.init.calculate_gain("relu")
                #)
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            else:
                fan_in = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.in_channels
                )
                nn.init.normal_(
                    m.weight, 0, torch.sqrt(torch.tensor(1.0 / fan_in))
                )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class SequentialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SequentialConvLayer, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_channels, 1, 1) for _ in range(out_channels)]
        )

    def forward(self, x):
        # Size of the input tensor
        batch_size, _, depth, height, width = x.size()

        # Initialize the output cubes
        outB = -10000 * torch.ones(batch_size, 1, depth, height, width).to(
            x.device
        )
        outC = torch.zeros(batch_size, 1, depth, height, width).to(x.device)

        for i, conv in enumerate(self.convs):
            # Apply the current filter
            outA = conv(x)

            # Find where the new filter gives a greater response than the max so far
            greater = outA > outB
            greater = greater.float()

            # Update outB with the max values so far
            outB = (1 - greater) * outB + greater * outA

            # Update outC with the index of the filter with the max response so far
            outC = (1 - greater) * outC + greater * i

        return outC


class MeshNet(nn.Module):
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(self, in_channels, n_classes, channels, config_file, fat=None, affine=False):
        """Init"""
        self.affine = affine
        with open(config_file, "r") as f:
            config = set_channel_num(
                json.load(f), in_channels, n_classes, channels
            )

        if fat is not None:
            chn = int(channels * 1.5)
            if fat in {"i", "io"}:
                config["layers"][0]["out_channels"] = chn
                config["layers"][1]["in_channels"] = chn
            if fat == "io":
                config["layers"][-1]["in_channels"] = chn
                config["layers"][-2]["out_channels"] = chn
            if fat == "b":
                config["layers"][3]["out_channels"] = chn
                config["layers"][4]["in_channels"] = chn

        super(MeshNet, self).__init__()

        layers = [
            construct_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                affine=self.affine,
                # with layer-norm we need no bias as we z-score channels anyway
                **{**block_kwargs, "bias": False},  # **block_kwargs,
            )
            for block_kwargs in config["layers"]
        ]
        # layers[-1] = SequentialConvLayer(
        #    layers[-1][0].in_channels, layers[-1][0].out_channels
        # )
        layers[-1] = layers[-1][0]
        self.model = nn.Sequential(*layers)
        # Add bias to the last layer
        self.model[-1].bias = nn.Parameter(torch.zeros(self.model[-1].out_channels))
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x


class FusedMeshNet(nn.Module):
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
    ):
        """Init"""
        with open(config_file, "r") as f:
            config = set_channel_num(
                json.load(f), in_channels, n_classes, channels
            )

        super(FusedMeshNet, self).__init__()

        layers = [
            construct_fusedlayer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                **block_kwargs,
            )
            for block_kwargs in config["layers"]
        ]
        # layers[-1] = SequentialConvLayer(layers[-1][0].in_channels, layers[-1][0].out_channels)
        layers[-1] = layers[-1][0]
        self.model = nn.Sequential(*layers)
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x


class MixedMeshNet(nn.Module):
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        highreslayers,
        fat=None,
    ):
        """Init"""
        with open(config_file, "r") as f:
            config = set_channel_num(
                json.load(f), in_channels, n_classes, channels
            )

        if fat is not None:
            chn = int(channels * 1.5)
            if fat in {"i", "io"}:
                config["layers"][0]["out_channels"] = chn
                config["layers"][1]["in_channels"] = chn
            if fat == "io":
                config["layers"][-1]["in_channels"] = chn
                config["layers"][-2]["out_channels"] = chn
            if fat == "b":
                config["layers"][3]["out_channels"] = chn
                config["layers"][4]["in_channels"] = chn

        super(MixedMeshNet, self).__init__()

        layers = [
            construct_mixedlayer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                highreslayers=highreslayers,
                **block_kwargs,
            )
            for block_kwargs in config["layers"]
        ]
        # layers[-1] = SequentialConvLayer(layers[-1][0].in_channels, layers[-1][0].out_channels)
        layers[-1] = layers[-1][0]
        self.model = nn.Sequential(*layers)
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x


class CheckpointMixin:
    def train_forward(self, x):
        if not getattr(self, "use_checkpoint", True):
            return self.model(x)
        y = x
        y.requires_grad_()
        n_layers = len(self.model)
        # checkpoint_segments controls the recompute granularity:
        #   None / <=0  -> one segment per layer (max memory saving, max recompute
        #                  + kernel-launch overhead; the original behavior)
        #   k           -> split the trunk into k segments, recompute one segment
        #                  at a time. Fewer, larger segments => less recompute and
        #                  fewer kernel launches at the cost of higher peak memory.
        segments = getattr(self, "checkpoint_segments", None)
        if not segments or segments < 1:
            segments = n_layers

        # Partial checkpointing: run the first `keep` layers normally (their
        # activations are RETAINED, so they are NOT recomputed in backward) and
        # checkpoint only the remaining suffix. This spends spare GPU memory to
        # cut the recompute tax (the dominant cost on the deep model). keep=0
        # (default) reproduces full checkpointing exactly. The slices are local
        # (not assigned to self), so they share the registered layer modules and
        # do NOT double-register parameters.
        keep = int(getattr(self, "checkpoint_keep_layers", 0) or 0)
        keep = max(0, min(keep, n_layers))
        if keep > 0:
            head = self.model[:keep]
            tail = self.model[keep:]
            y = head(y)
            if len(tail) > 0:
                seg = min(int(segments), len(tail))
                y = checkpoint_sequential(
                    tail, seg, y, preserve_rng_state=False, use_reentrant=False
                )
            return y

        segments = min(int(segments), n_layers)
        y = checkpoint_sequential(
            self.model, segments, y, preserve_rng_state=False, use_reentrant=False
        )
        return y

    def eval_forward(self, x):
        """Forward pass"""
        self.model.eval()
        with torch.inference_mode():
            x = self.model(x)
        return x

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


class enMesh_checkpoint(CheckpointMixin, MeshNet):
    pass


class xenMesh_checkpoint(CheckpointMixin, MixedMeshNet):
    pass


class fenMesh_checkpoint(CheckpointMixin, FusedMeshNet):
    pass


class SpatialAEMeshNet(CheckpointMixin, nn.Module):
    """MeshNet dilated trunk wrapped in a peak-memory-neutral spatial
    autoencoder bottleneck, with NO skip connections (anti-U-Net).

    Data flow (R = full volume edge, e.g. 256):

        in(1ch @ R^3)
          -> stem    : Conv3 1->C            @ R^3      (full-res features)
          -> down    : R^3 -> (R/2)^3, C->Cb            (avgpool+1x1, or strided)
          -> trunk   : the dilated stack, Cb->Cb @ (R/2)^3   (the heavy compute)
          -> up      : (R/2)^3 -> R^3, Cb->C            (transposed conv, or trilinear+conv)
          -> refine  : Conv3 C->C            @ R^3      (full-res detail recovery)
          -> head    : Conv1 C->n_classes    @ R^3
        out(n_classes @ R^3)

    Why this is peak-memory-neutral vs the flat trunk: the wide (Cb-channel)
    dilated stack runs at (R/2)^3 = 1/8 the voxels, so its activations are
    ~Cb/8 = C/4 of a single flat-trunk layer buffer. The only full-res
    activations are the few C-channel stem/up/refine/head maps, each ~ one
    flat-trunk layer. There are NO skips, so nothing is retained across the
    bottleneck. Net peak ~= the flat model's peak (one C x R^3 buffer).

    Why a bottleneck helps when TRAINING on SynthSeg (vs the flat trunk that
    "gets the brain but misses the folds"): SynthSeg randomizes per-label
    intensity every sample, so segmentation must come from geometry, not
    intensity. avgpool downsampling averages out that per-label intensity
    noise, and at (R/2)^3 a dense rate-1 3x3 already spans fold-scale extent
    so the trunk reasons about shape without needing the largest dilations;
    the learned transposed-conv upsample acts as a shape prior that
    reconstructs the folded cortical ribbon.

    Implementation note: ``self.model`` is a single flat ``nn.Sequential`` and
    this class inherits ``CheckpointMixin``, so gradient checkpointing
    (``use_checkpoint`` / ``checkpoint_segments`` / ``checkpoint_keep_layers``),
    ``channels_last_3d`` and the train/eval forward contract behave EXACTLY as
    for the flat ``enMesh_checkpoint(_gn)`` model -- the trainer needs no
    special handling and calls ``model(sample) -> logits`` as usual.
    """

    def __init__(self, in_channels, n_classes, channels, config_file,
                 affine=False, bottleneck_mult=2, downsample="avgpool",
                 upsample="transposed"):
        super().__init__()
        C = int(channels)
        Cb = int(channels) * int(bottleneck_mult)
        with open(config_file, "r") as f:
            cfg = json.load(f)
        gelu = cfg.get("gelu", False)
        bnorm = cfg.get("bnorm", True)
        dropout_p = cfg.get("dropout_p", 0)
        act = (lambda: nn.GELU()) if gelu else (lambda: nn.ReLU(inplace=True))

        def block(cin, cout, k, pad, stride=1, dil=1):
            # same conv+GroupNorm+act block factory the flat trunk uses
            return construct_layer(
                dropout_p=0, bnorm=bnorm, gelu=gelu, affine=affine,
                in_channels=cin, out_channels=cout, kernel_size=k,
                padding=pad, stride=stride, dilation=dil, bias=False,
            )

        layers = []
        # --- full-res stem: 1 -> C ---
        layers.append(block(in_channels, C, k=3, pad=1))

        # --- downsample R -> R/2, C -> Cb ---
        if downsample == "strided":
            layers.append(block(C, Cb, k=2, pad=0, stride=2))
        else:  # "avgpool": anti-aliased (averages SynthSeg intensity noise) + 1x1 expand
            layers.append(nn.AvgPool3d(kernel_size=2, stride=2))
            layers.append(block(C, Cb, k=1, pad=0))

        # --- dilated trunk at R/2, Cb -> Cb (reuse the config_file schedule) ---
        # Built with in=n_classes=channels=Cb so every layer (incl. the first
        # and final 1x1) operates at width Cb; we splice in its conv blocks.
        trunk = MeshNet(in_channels=Cb, n_classes=Cb, channels=Cb,
                        config_file=config_file, affine=affine)
        layers.extend(list(trunk.model))

        # --- upsample R/2 -> R, Cb -> C ---
        if upsample == "transposed":
            layers.append(nn.ConvTranspose3d(Cb, C, kernel_size=2, stride=2,
                                             bias=False))
        else:  # "trilinear" + 3x3 conv
            layers.append(nn.Upsample(scale_factor=2, mode="trilinear",
                                      align_corners=False))
            layers.append(nn.Conv3d(Cb, C, kernel_size=3, padding=1, bias=False))
        if bnorm:
            layers.append(nn.GroupNorm(num_groups=C, num_channels=C,
                                       affine=affine))
        layers.append(act())
        if dropout_p > 0:
            layers.append(nn.Dropout3d(dropout_p))

        # --- full-res refine: C -> C ---
        layers.append(block(C, C, k=3, pad=1))

        # --- head: C -> n_classes (1x1, with bias; no norm/act) ---
        head = nn.Conv3d(C, n_classes, kernel_size=1)
        layers.append(head)

        self.model = nn.Sequential(*layers)
        init_weights(self.model)
        # bias on the logit head (matches flat MeshNet's final-layer treatment)
        if self.model[-1].bias is not None:
            nn.init.constant_(self.model[-1].bias, 0.0)


# class enMesh_checkpoint(MeshNet):
#     def train_forward(self, x):
#         y = x
#         y.requires_grad_()
#         y = checkpoint_sequential(
#             self.model, len(self.model), y, preserve_rng_state=False
#         )
#         return y

#     def eval_forward(self, x):
#         """Forward pass"""
#         self.model.eval()
#         with torch.inference_mode():
#             x = self.model(x)
#         return x

#     def forward(self, x):
#         if self.training:
#             return self.train_forward(x)
#         else:
#             return self.eval_forward(x)


# class xenMesh_checkpoint(MixedMeshNet):
#     def train_forward(self, x):
#         y = x
#         y.requires_grad_()
#         y = checkpoint_sequential(
#             self.model, len(self.model), y, preserve_rng_state=False
#         )
#         return y

#     def eval_forward(self, x):
#         """Forward pass"""
#         self.model.eval()
#         with torch.inference_mode():
#             x = self.model(x)
#         return x

#     def forward(self, x):
#         if self.training:
#             return self.train_forward(x)
#         else:
#             return self.eval_forward(x)


# class fenMesh_checkpoint(FusedMeshNet):
#     def train_forward(self, x):
#         y = x
#         y.requires_grad_()
#         y = checkpoint_sequential(
#             self.model, len(self.model), y, preserve_rng_state=False
#         )
#         return y

#     def eval_forward(self, x):
#         """Forward pass"""
#         self.model.eval()
#         with torch.inference_mode():
#             x = self.model(x)
#         return x

#     def forward(self, x):
#         if self.training:
#             return self.train_forward(x)
#         else:
#             return self.eval_forward(x)


class enMesh(MeshNet):
    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        optimize_inline=False,
    ):
        super(enMesh, self).__init__(
            in_channels, n_classes, channels, config_file
        )
        self.n_classes = n_classes
        self.optimize_inline = optimize_inline
        if self.optimize_inline:
            self.optimizers = [
                torch.optim.Adam(net.parameters(), lr=0.02)
                for net in self.model
            ]

    def get_grads(self, grads):
        def show(self, grad_input, grad_output):
            grads["in"] = grad_input
            grads["out"] = grad_output

        return show

    def set_requires_grad_layer(self, layer, flag, trainBN=True):
        layer.train(flag)
        for x in layer.parameters():
            if not flag:
                del x.grad
                x.detach()
            x.grad = [None, x.grad][flag]
            x.requires_grad = flag
        if (
            trainBN
            and isinstance(layer, torch.nn.Sequential)
            and isinstance(layer[1], torch.nn.BatchNorm3d)
        ):
            layer[1].training = True
            layer[1].requires_grad = True

    def unset_grad(self, layer):
        self.set_requires_grad_layer(layer, False)

    def set_grad(self, layer):
        self.set_requires_grad_layer(layer, True)

    def dump_tensors(gpu_only=True):
        # torch.cuda.empty_cache()
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if not gpu_only or obj.is_cuda:
                        del obj
                        gc.collect()
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        del obj
                        gc.collect()
            except Exception as e:
                pass

    def eval_forward(self, x):
        """Forward pass"""
        with torch.inference_mode():
            for i, layer in enumerate(self.model):
                x = layer(x)
        return x

    def forward(self, x, y=None, loss=None, verbose=False):
        if self.training:
            return self.backforward(x, y, loss, verbose=verbose)
        else:
            return self.eval_forward(x)

    def backforward(self, x, y, loss, verbose=False):
        if verbose:
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"total    : {info.total}")
            print(f"free     : {info.free}")
            print(f"used     : {info.used}")
            print(f"used fr  : {info.used/info.total}")

        gradients = {}
        layers = [p for p in self.model]
        for p in layers:
            self.unset_grad(p)

        grads = {}
        handle = layers[-1].register_full_backward_hook(self.get_grads(grads))

        self.set_grad(layers[-1])
        input = x
        input.requires_grad = False
        for count, layer in enumerate(layers):
            input = layer(input)
        y_hat = input
        input.requires_grad_()
        input.detach()

        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"used fr  : {info.used/info.total}")

        if isinstance(loss, torch.nn.CrossEntropyLoss):
            output = loss(input, y)
        else:
            one_hot_targets = torch.nn.functional.one_hot(
                y, self.n_classes
            ).permute(0, 4, 1, 2, 3)
            logits_softmax = F.softmax(input, dim=1)
            output = loss(logits_softmax, one_hot_targets)
        output.backward()
        output.detach()
        lss_value = output
        del output
        del input
        self.unset_grad(layers[-1])
        handle.remove()

        dloss_dx2 = grads["out"][0]

        del grads["in"]

        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"used fr  : {info.used/info.total}")
            print("*" * 20)

        # unembedded = True
        for i in range(len(layers) - 1, -1, -1):
            input = x.detach().clone()
            input.requires_grad = False
            grads = {}
            handle = layers[i].register_full_backward_hook(
                self.get_grads(grads)
            )
            self.set_grad(layers[i])

            # Recompute the forward pass up to the current layer
            for j in range(0, i + 1):
                if j == i:
                    input.detach()
                    input.requires_grad_()
                input = layers[j](input)

            input.detach()
            torch.autograd.backward(input, dloss_dx2)

            del dloss_dx2
            dloss_dx2 = grads["in"][0]

            if self.optimize_inline:
                self.optimizers[i].step()
                self.optimizers[i].zero_grad(set_to_none=True)
            else:
                gradients[i] = [x.grad for x in layers[i].parameters()]

            self.unset_grad(layers[i])
            handle.remove()
            del input.grad
            del x.grad
            del input
            x.requires_grad = False
        del dloss_dx2
        self.model.eval()
        if not self.optimize_inline:
            for i in range(len(layers)):
                # self.set_grad(layers[i])
                for p, g in zip(layers[i].parameters(), gradients[i]):
                    p.grad = g
        del layers
        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"{i} used fr  : {info.used/info.total}")
        # torch.cuda.empty_cache()
        # self.dump_tensors()

        return lss_value, y_hat


class MeshNet_fad(MeshNet):
    """MeshNet with forward AD"""

    def __init__(self, in_channels, n_classes, channels, config_file):
        """Init"""
        super(MeshNet_fad, self).__init__(
            in_channels, n_classes, channels, config_file
        )
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        (
            self.loss_func,
            self.loss_params,
            self.loss_buffers,
        ) = make_functional_with_buffers(
            self.loss, disable_autograd_tracking=True
        )

    def eval_forward(self, x):
        """Forward pass"""
        with torch.inference_mode():
            for i, layer in enumerate(self.model):
                x = layer(x)
        return x

    def forward(self, x, y=None, loss=None, verbose=False):
        if self.training:
            return self.forwardforward(x)
        else:
            return self.eval_forward(x)

    def layergrads(self, layer, dotrain):
        for param in layer.parameters():
            param.requires_grad = dotrain
            param.grad = None

    def forwardforward(self, x, y):
        x.requires_grad = False
        grads = {}
        jvps = None
        for idx, layer in enumerate(self.model):
            func, params, buffers = make_functional_with_buffers(
                layer, disable_autograd_tracking=True
            )

            def func_params_only(params):
                return func(params, buffers, x)

            def func_values_only(x):
                return func(params, buffers, x)

            def f(x, tangent):
                return torch.func.jvp(func_values_only, (x,), (tangent,))

            # Create random vector from spherical Gaussian normalized to length 1
            tangents = tuple(
                [
                    v / v.norm()
                    for p in params
                    for v in [torch.randn_like(p, requires_grad=False)]
                ]
            )
            for p, g in zip(layer.parameters(), tangents):
                p.grad = g
            # Compute layer output
            output, jvp_out = torch.func.jvp(
                func_params_only, (params,), (tangents,)
            )
            output.detach()
            jvp_out.detach()

            jvp_out = torch.unsqueeze(jvp_out, dim=0)
            if jvps is not None:
                newout, jacs = torch.func.vmap(f)(
                    torch.stack((x,) * jvps.shape[0]), jvps
                )
                jvps = torch.cat((jacs, jvp_out), 0)
                del jacs
                del newout
                del jvp_out
            else:
                jvps = jvp_out
                jvps.detach()
                del jvp_out

            x = output
            del output
            del func
            del params
            del buffers

        def loss_values_only(x):
            return self.loss_func(self.loss_params, self.loss_buffers, x, y)

        def loss_f(x, tangent):
            return torch.func.jvp(loss_values_only, (x,), (tangent,))

        newout, jacs = torch.func.vmap(loss_f)(
            torch.stack((x,) * jvps.shape[0]), jvps
        )
        for layer, jvps in zip(self.model, jacs):
            for p in layer.parameters():
                p.grad *= jvps
        del jacs
        del newout
        del x
        # gc.collect()
        # torch.cuda.empty_cache()
        print("final:   ", torch.cuda.memory_allocated())
        return True


class bpfreeMesh(MeshNet):
    """ """

    def __init__(
        self, in_channels, n_classes, channels, config_file, samples=10
    ):
        """Init"""
        super(bpfreeMesh, self).__init__(
            in_channels, n_classes, channels, config_file
        )
        self.samples = samples
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        (
            self.loss_func,
            self.loss_params,
            self.loss_buffers,
        ) = make_functional_with_buffers(
            self.loss, disable_autograd_tracking=True
        )

    def eval_forward(self, x):
        """Forward evaluation pass"""
        with torch.inference_mode():
            for i, layer in enumerate(self.model):
                x = layer(x)
        return x

    def forward(self, x, y=None, loss=None, verbose=False):
        if self.training:
            return self.forwardforward(x)
        else:
            return self.eval_forward(x)

    def layergrads(self, layer, dotrain):
        for param in layer.parameters():
            param.requires_grad = dotrain
            param.grad = None

    def forwardforward(self, x, y):
        x.requires_grad = False
        grads = {}
        jvps = None
        for idx, layer in enumerate(self.model):
            func, params, buffers = make_functional_with_buffers(
                layer, disable_autograd_tracking=True
            )

            def func_params_only(params):
                return func(params, buffers, x)

            def p(x, tangent):
                return torch.func.jvp(func_params_only, (x,), (tangent,))

            def func_values_only(x):
                return func(params, buffers, x)

            def f(x, tangent):
                return torch.func.jvp(func_values_only, (x,), (tangent,))

            # Create random vector from spherical Gaussian N(0,1)
            tangents = tuple(
                [
                    v
                    for p in params
                    for v in [
                        torch.randn(self.samples, *p.shape, requires_grad=False)
                    ]
                ]
            )

            # Compute layer output at all samples
            output, params_tangents = torch.func.vmap(p)(
                (x.repeat(self.samples, 1) for x in params), tangents
            )
            print("output shape: ", output.shape)
            print("params shape: ", params_tangents.shape)

            output.detach()
            params_tangents.detach()

            jvp_out = torch.unsqueeze(jvp_out, dim=0)
            if jvps is not None:
                newout, jacs = torch.func.vmap(f)(
                    torch.stack((x,) * jvps.shape[0]), jvps
                )
                jvps = torch.cat((jacs, jvp_out), 0)
                del jacs
                del newout
                del jvp_out
            else:
                jvps = jvp_out
                jvps.detach()
                del jvp_out

            x = output
            del output
            del func
            del params
            del buffers

        def loss_values_only(x):
            return self.loss_func(self.loss_params, self.loss_buffers, x, y)

        def loss_f(x, tangent):
            return torch.func.jvp(loss_values_only, (x,), (tangent,))

        newout, jacs = torch.func.vmap(loss_f)(
            torch.stack((x,) * jvps.shape[0]), jvps
        )
        for layer, jvps in zip(self.model, jacs):
            for p in layer.parameters():
                p.grad *= jvps
        del jacs
        del newout
        del x
        # gc.collect()
        # torch.cuda.empty_cache()
        print("final:   ", torch.cuda.memory_allocated())
        return True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())  # if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = 21
    cubesize = 256
    classes = 104
    batch = 1
    config_file = "modelAE.json"

    # emodel = bpfreeMesh(1, classes, channels, config_file).to(device)
    emodel = enMesh_checkpoint(1, classes, channels, config_file).to(device)
    print(emodel)
    stop
    # emodel = torch.compile(emodel)

    num_params = count_parameters(emodel)
    print(f"Number of parameters: {num_params}")

    x = torch.rand(batch, 1, *(cubesize,) * 3, requires_grad=False).to(device)
    y = torch.randint(
        0, classes, (batch, *(cubesize,) * 3), requires_grad=False
    ).to(device)
    # model.eval()
    # t0 = time.time()
    # for i in range(10):
    # r = model.forwardforward(x, y)
    #    r = model.forward(x)
    # t1 = time.time()
    # print(t1-t0)
    print("enmesh")
    from blendbatchnorm import fuse_bn_recursively

    emodel = fuse_bn_recursively(emodel)
    criterion = torch.nn.CrossEntropyLoss()
    emodel.train(False)
    t0 = time.time()
    # x.requires_grad = True
    # with torch.no_grad():
    with torch.inference_mode():
        for i in range(10):
            r = emodel.forward(x)  # forwardforward(x, y)
            del r
            torch.cuda.empty_cache()
        # loss = criterion(r, y)
        # loss.backward()
    t1 = time.time()
    print(t1 - t0)


class TowerMeshNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        total_channels,
        config_files,
    ):
        super().__init__()
        channels = int(total_channels) // len(config_files)
        configs = {}
        towers = {}
        for i, config_file in enumerate(config_files):
            with open(config_file, "r") as f:
                config = set_channel_num(
                    json.load(f), in_channels, n_classes, channels
                )
            configs[i] = config
            layers = [
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config["gelu"],
                    **block_kwargs,
                )
                for block_kwargs in config["layers"]
            ]
            layers[-1] = nn.Identity()
            towers[f"tower_{i}"] = nn.Sequential(*layers)
        self.towers = nn.ModuleDict(towers)
        self._last_layer = nn.Conv3d(
            total_channels,
            n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

    def forward(self, x):
        """Forward pass"""
        y = []
        for tower in self.towers.values():
            y.append(tower(x))
        x = torch.cat(y, dim=1)
        x = self._last_layer(x)
        return x
