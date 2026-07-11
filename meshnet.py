import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.utils.checkpoint import checkpoint_sequential
import json
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


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


def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
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
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def init_weights(model):
    """Set weights to be xavier normal for all Convs"""
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

# This is the base model but if used just as is it consumes too much memory
class MeshNet(nn.Module):
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(self, in_channels, n_classes, channels, config_file):
        """Init"""
        with open(config_file, "r") as f:
            config = set_channel_num(
                json.load(f), in_channels, n_classes, channels
            )
        super(MeshNet, self).__init__()

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
        self.model = nn.Sequential(*layers)
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x

# However, the model is fully sequential and the class below takes
# advantage of this fact greatly reducing memory used at training
# Note, at inference, the model supposed to automatically only occupy
# a single layer worth of GPU memory
class enMesh_checkpoint(MeshNet):
    def train_forward(self, x):
        if not getattr(self, "use_checkpoint", True):
            return self.model(x)
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            self.model, len(self.model), y, preserve_rng_state=False, use_reentrant=False
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


class enMesh_fixedpoint(enMesh_checkpoint):
    def train_forward(self, x):
        x.requires_grad_()
        for i in range(self.max_iter):
            y = checkpoint_sequential(
                self.model, len(self.model), y, preserve_rng_state=False, use_reentrant=False
            )
            # need to prepend on the channel dimension a tensor x with self.channels as number of channels + N for the y channels
            x = torch.cat([x, y], dim=1)
        return y
# The above is of course a classical trade of computation for memory.  However, pytorch still caches a lot and thus uses more memory than needed. Below is my manual implementation that is slower than above, but most memory economical
#
# Why am I giving this slower model to you :) If you want to train on GPUs with 40GB or 48GB (e.g. A40) then the model below is helpful. I suspect this is not meshnet specific and u-net is also positively affected by what I am going to describe, but here is how I train meshnets.  Given training data and model architecture (mostly number of channels).
#
# 1. run a few epochs 10-20 on 32x32x32 cubesize. When traning on real data it usually get's me DICE of 0.1-0.2 depending on difficulty of the data. Use OnceCycleLR for best results
# 2. repear the above using the same model weights in a curriculum of 64, 96, 128, 192, 256 cubesize 256 will not fit in 40Gb, so you will use the class below (the checkpoint and the architecture is the same)
# 3. then you can do some more tuning but just on 256 cube size.
#
# The trained model can be applied to tensors of any size, not necessarily 256 cube, but the voxel should be isotropic 1mm or it fails terribly.
#
# Interestingly, when trained on your generator, it can bootsrap from cubesize 128, no need to train on low cube sizes.
        
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


class ChannelSEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps.

    Ref: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

    Adds channel-wise attention whose parameters live entirely in
    Linear layers operating on [B, hidden] vectors.  No additional
    feature maps of full spatial size are ever allocated, so peak
    GPU memory stays identical to the base convolution.

    The excitation MLP can be made arbitrarily wide (`hidden_channels`)
    and deep (`num_fc_layers`) to pack more learnable capacity without
    any spatial-memory cost.

    Args:
        channels: number of input/output channels (C).
        hidden_channels: explicit width of every hidden FC layer.
            When set, `reduction` and `min_channels` are ignored.
            Use this to decouple SE capacity from C.
        reduction: reduction ratio applied to C (default: 4).
            Only used when `hidden_channels` is None.
        min_channels: floor on bottleneck width when using
            `reduction` (default: 4).
        num_fc_layers: total number of Linear layers in the
            excitation path (default: 2, i.e. classic SE).
            Values > 2 insert extra hidden→hidden layers.
    """

    def __init__(self, channels, hidden_channels=None, reduction=4,
                 min_channels=4, num_fc_layers=2):
        super().__init__()
        if hidden_channels is not None:
            bottleneck = hidden_channels
        else:
            bottleneck = max(channels // reduction, min_channels)

        self.squeeze = nn.AdaptiveAvgPool3d(1)

        # Build excitation MLP: in→hidden [→hidden …] →out→σ
        layers = []
        in_feat = channels
        for _ in range(num_fc_layers - 1):
            layers.append(nn.Linear(in_feat, bottleneck, bias=True))
            layers.append(nn.ReLU(inplace=True))
            in_feat = bottleneck
        layers.append(nn.Linear(in_feat, channels, bias=True))
        layers.append(nn.Sigmoid())
        self.excitation = nn.Sequential(*layers)
        
        # Initialize final layer to 0 so the block acts as an identity at first
        nn.init.zeros_(self.excitation[-2].weight)
        nn.init.zeros_(self.excitation[-2].bias)

    def forward(self, x):
        b, c = x.shape[:2]
        # Squeeze: global average pool over (D, H, W) → [B, C]
        scale = self.squeeze(x).view(b, c)
        # Excitation: MLP → [B, C]
        scale = self.excitation(scale)
        # Scale: broadcast-multiply back onto spatial feature map (x2 to preserve variance)
        return x * scale.view(b, c, 1, 1, 1) * 2.0


def _inject_se_blocks(model, se_hidden_channels=None, se_reduction=4,
                      se_min_channels=4, se_num_fc_layers=2):
    """Inject SE blocks after each Conv+BN+ReLU sequential layer.

    Leaves bare Conv3d layers (e.g. the final 1×1 output conv) untouched.
    """
    new_layers = []
    for layer in model:
        if isinstance(layer, nn.Sequential):
            out_ch = layer[0].out_channels
            se = ChannelSEBlock3D(
                out_ch,
                hidden_channels=se_hidden_channels,
                reduction=se_reduction,
                min_channels=se_min_channels,
                num_fc_layers=se_num_fc_layers,
            )
            layer = nn.Sequential(*list(layer.children()), se)
        new_layers.append(layer)
    return nn.Sequential(*new_layers)


class enMesh_checkpoint_SE(enMesh_checkpoint):
    """MeshNet with Squeeze-and-Excitation channel attention.

    Inherits checkpoint-sequential training from enMesh_checkpoint.
    SE blocks are appended after each Conv+BN+ReLU block (every layer
    except the final 1×1 classification conv).

    Memory overhead: none beyond base MeshNet — SE operates on [B, C]
    and [B, hidden] vectors only.  Peak spatial activation memory is
    identical to a single MeshNet layer.

    Parameter budget is controlled independently of memory via:
      - hidden_channels: width of the excitation MLP (can be >> C)
      - num_fc_layers:   depth of the excitation MLP (≥ 2)

    Example parameter counts (9 SE blocks):
      C=5,  hidden=64, layers=2:   9 × 2×5×64       =  5,760
      C=5,  hidden=64, layers=3:   9 × (5×64+64²+64×5) = 42,624
      C=21, hidden=64, layers=2:   9 × 2×21×64      = 24,192
    """

    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        se_hidden_channels=None,
        se_reduction=4,
        se_min_channels=4,
        se_num_fc_layers=2,
    ):
        super().__init__(in_channels, n_classes, channels, config_file)
        self.model = _inject_se_blocks(
            self.model, se_hidden_channels, se_reduction,
            se_min_channels, se_num_fc_layers,
        )


class enMesh_SE(enMesh):
    """MeshNet-SE with manual layer-by-layer backprop for large volumes.

    Inherits the memory-minimal manual backprop from enMesh.
    SE blocks are injected identically to enMesh_checkpoint_SE.
    """

    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        optimize_inline=False,
        se_hidden_channels=None,
        se_reduction=4,
        se_min_channels=4,
        se_num_fc_layers=2,
    ):
        super().__init__(
            in_channels, n_classes, channels, config_file, optimize_inline
        )
        self.model = _inject_se_blocks(
            self.model, se_hidden_channels, se_reduction,
            se_min_channels, se_num_fc_layers,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = 5
    cubesize = 256
    classes = 3
    batch = 1
    config_file = "modelAE.json"

    emodel = enMesh_checkpoint(1, classes, channels, config_file).to(device)
