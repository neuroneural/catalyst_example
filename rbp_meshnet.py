"""Recurrent backpropagation MeshNet experiment.

This model solves a hidden-state fixed point

    h* = Trunk(concat(input, h*))

with Anderson acceleration in the forward pass and RBP/implicit gradients in
the backward pass. The logits head is applied once after convergence.
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from refiner import set_channel_num, init_weights, _anderson_forward


def construct_layer(dropout_p=0, bnorm=True, gelu=False, groupnorm=False, **kwargs):
    layers = [nn.Conv3d(**kwargs)]
    out_channels = kwargs["out_channels"]
    if bnorm:
        if groupnorm:
            layers.append(nn.GroupNorm(out_channels, out_channels, affine=False))
        else:
            layers.append(nn.BatchNorm3d(out_channels, track_running_stats=True))
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


class RBPTrunkStep(nn.Module):
    def __init__(self, trunk, relax=0.2):
        super().__init__()
        self.trunk = trunk
        self.relax = relax

    @property
    def alpha(self):
        return torch.ones((), device=next(self.parameters()).device)

    def forward(self, h, x, _unused=None):
        z = torch.cat([x, h], dim=1)
        proposal = checkpoint_sequential(
            self.trunk,
            len(self.trunk),
            z,
            preserve_rng_state=False,
            use_reentrant=False,
        )
        return (1.0 - self.relax) * h + self.relax * proposal


class RBPStateSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h0, step_module, max_iter, tol, anderson_m, bwd_iter, bwd_tol, bwd_m, *params):
        ctx.step_module = step_module
        ctx.bwd_iter = bwd_iter
        ctx.bwd_tol = bwd_tol
        ctx.bwd_m = bwd_m
        ctx.num_params = len(params)

        with torch.no_grad():
            if anderson_m > 0:
                h_star = _anderson_forward(step_module, h0, x, max_iter, tol, anderson_m)
            else:
                h = h0
                step_module.last_fwd_iters = max_iter
                step_module.last_fwd_rel = 0.0
                for i in range(max_iter):
                    h_next = step_module(h, x)
                    if i > 0:
                        rel = (h_next - h).norm() / (h.norm() + 1e-8)
                        step_module.last_fwd_iters = i + 1
                        step_module.last_fwd_rel = rel.item()
                        if rel < tol:
                            break
                    h = h_next
                h_star = h
            step_module.last_fwd_delta = (h_star - h0).norm().div(h_star.norm() + 1e-8).item()
        ctx.save_for_backward(x.detach(), h_star.detach())
        return h_star

    @staticmethod
    def backward(ctx, grad_output):
        x, h_star = ctx.saved_tensors
        step_module = ctx.step_module

        def vjp_h(v):
            with torch.enable_grad():
                h_in = h_star.detach().requires_grad_(True)
                x_in = x.detach().requires_grad_(True)
                out = step_module(h_in, x_in)
                grad_h = torch.autograd.grad(out, h_in, v, retain_graph=False, create_graph=False)[0]
            return grad_h

        # Solve v = grad + J_G^T v with fixed-point iteration. This is the
        # RBP adjoint solve; each iteration checkpoints the trunk VJP.
        v = grad_output
        for _ in range(ctx.bwd_iter):
            v_next = grad_output + vjp_h(v)
            rel = (v_next - v).norm() / (v.norm() + 1e-8)
            v = v_next
            if rel < ctx.bwd_tol:
                break

        with torch.enable_grad():
            h_in = h_star.detach().requires_grad_(True)
            x_in = x.detach().requires_grad_(True)
            params = tuple(step_module.parameters())
            out = step_module(h_in, x_in)
            grads = torch.autograd.grad(
                out,
                (x_in, h_in) + params,
                v,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
        grad_x = grads[0]
        grad_params = tuple(torch.zeros_like(p) if g is None else g for g, p in zip(grads[2:], params))
        return (
            grad_x,
            None,
            None,
            None, None, None, None, None, None,
            *grad_params,
        )


class RBPMeshNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        groupnorm=False,
        rbp_max_iter=30,
        rbp_tol=1e-4,
        rbp_anderson_m=5,
        rbp_backward_max_iter=30,
        rbp_backward_tol=1e-4,
        rbp_backward_anderson_m=0,
        rbp_relax=0.2,
        **_,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.rbp_max_iter = rbp_max_iter
        self.rbp_tol = rbp_tol
        self.rbp_anderson_m = rbp_anderson_m
        self.rbp_backward_max_iter = rbp_backward_max_iter
        self.rbp_backward_tol = rbp_backward_tol
        self.rbp_backward_anderson_m = rbp_backward_anderson_m
        self.rbp_relax = rbp_relax

        with open(config_file, "r") as f:
            config = set_channel_num(json.load(f), in_channels, n_classes, channels)

        trunk_blocks = config["layers"][:-1]
        trunk_blocks[0] = dict(trunk_blocks[0])
        trunk_blocks[0]["in_channels"] = in_channels + channels
        trunk_blocks[0]["out_channels"] = channels

        self.trunk = nn.Sequential(*[
            construct_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                groupnorm=groupnorm,
                **block_kwargs,
            )
            for block_kwargs in trunk_blocks
        ])
        final_kwargs = dict(config["layers"][-1])
        self.head = nn.Conv3d(**final_kwargs)
        self.step = RBPTrunkStep(self.trunk, relax=rbp_relax)
        init_weights(self)

    def solve_state(self, x):
        h0 = torch.zeros(x.shape[0], self.channels, *x.shape[2:], device=x.device, dtype=x.dtype)
        if self.training:
            h_star = RBPStateSolve.apply(
                x,
                h0,
                self.step,
                self.rbp_max_iter,
                self.rbp_tol,
                self.rbp_anderson_m,
                self.rbp_backward_max_iter,
                self.rbp_backward_tol,
                self.rbp_backward_anderson_m,
                *tuple(self.step.parameters()),
            )
            return h_star
        with torch.no_grad():
            if self.rbp_anderson_m > 0:
                h_star = _anderson_forward(self.step, h0, x, self.rbp_max_iter, self.rbp_tol, self.rbp_anderson_m)
                self.step.last_fwd_delta = (h_star - h0).norm().div(h_star.norm() + 1e-8).item()
                return h_star
            h = h0
            for _ in range(self.rbp_max_iter):
                h_next = self.step(h, x)
                if (h_next - h).norm() / (h.norm() + 1e-8) < self.rbp_tol:
                    return h_next
                h = h_next
            return h

    def forward(self, x, y=None, loss=None, verbose=False):
        h = self.solve_state(x)
        logits = self.head(h)
        if y is not None and loss is not None:
            return loss(logits, y), logits
        return logits


class enRBPMesh_checkpoint(RBPMeshNet):
    pass


class enRBPMesh(RBPMeshNet):
    def __init__(self, *args, optimize_inline=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimize_inline = optimize_inline
