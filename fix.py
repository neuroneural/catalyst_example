import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Tuple


# =============================================================================
# Core: the linear system solvers
# =============================================================================

def solve_fixed_point_backward_cg(
    g_fn: Callable,          # g(k, s) → k_next, the fixed-point map
    k_star: torch.Tensor,    # converged fixed point (detached)
    s: torch.Tensor,         # stats (detached)
    grad_output: torch.Tensor,  # ∂L/∂k* from upstream
    max_iter: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Solve (I - J_kᵀ) v = grad_output via conjugate gradient,
    where J_k = ∂g/∂k evaluated at (k*, s).
    
    Each CG iteration requires one VJP through g w.r.t. k.
    
    Returns v such that:
      ∂L/∂θ = v · ∂g/∂θ  (obtained via separate VJP call)
      ∂L/∂s = v · ∂g/∂s  (obtained via separate VJP call)
    """
    
    def matvec(v):
        """Compute (I - J_kᵀ) @ v without forming J_k."""
        with torch.enable_grad():
            k_in = k_star.detach().requires_grad_(True)
            g_out = g_fn(k_in, s.detach())
            Jt_v = torch.autograd.grad(
                outputs=g_out,
                inputs=k_in,
                grad_outputs=v,
                retain_graph=False,
                create_graph=False,
            )[0]
        if torch.isnan(Jt_v).any() or torch.isinf(Jt_v).any():
            Jt_v = torch.where(torch.isnan(Jt_v) | torch.isinf(Jt_v), torch.zeros_like(Jt_v), Jt_v)
        return v - Jt_v
    
    v = grad_output.clone()
    for i in range(max_iter):
        Av = matvec(v)
        Jt_v = v - Av  # J_kᵀ · v
        v_new = grad_output + Jt_v
        
        rel_change = (v_new - v).norm() / (v.norm() + 1e-8)
        v = v_new
        
        if rel_change < tol:
            break
    
    return v


def solve_fixed_point_backward_neumann(
    g_fn: Callable,
    k_star: torch.Tensor,
    s: torch.Tensor,
    grad_output: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Neumann series solver for (I - J_kᵀ)^{-1} · grad_output.
    
    v = Σ_{i=0}^{N} (J_kᵀ)^i · grad_output
    
    Computed iteratively:
      v_0 = grad_output
      v_{t+1} = grad_output + J_kᵀ · v_t
    """
    
    def vjp_Jk(v):
        """Compute J_kᵀ · v via one backward pass through g."""
        with torch.enable_grad():
            k_in = k_star.detach().requires_grad_(True)
            g_out = g_fn(k_in, s.detach())
            Jt_v = torch.autograd.grad(
                g_out, k_in, v,
                retain_graph=False,
                create_graph=False,
            )[0]
        if torch.isnan(Jt_v).any() or torch.isinf(Jt_v).any():
            Jt_v = torch.where(torch.isnan(Jt_v) | torch.isinf(Jt_v), torch.zeros_like(Jt_v), Jt_v)
        return Jt_v
    
    v = grad_output.clone()
    for i in range(max_iter):
        Jt_v = vjp_Jk(v)
        v_new = grad_output + Jt_v
        
        rel = (v_new - v).norm() / (v.norm() + 1e-8)
        v = v_new
        if rel < tol:
            break
    
    return v


def solve_fixed_point_backward_anderson(
    g_fn: Callable,
    k_star: torch.Tensor,
    s: torch.Tensor,
    grad_output: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-5,
    m: int = 3,
) -> torch.Tensor:
    """
    Anderson-accelerated Neumann series.
    
    The backward fixed-point iteration v_{t+1} = grad_output + J_kᵀ · v_t
    can itself be accelerated with Anderson mixing.
    """
    B = k_star.shape[0]
    
    def vjp_Jk(v):
        with torch.enable_grad():
            k_in = k_star.detach().requires_grad_(True)
            g_out = g_fn(k_in, s.detach())
            Jt_v = torch.autograd.grad(
                g_out, k_in, v,
                retain_graph=False,
                create_graph=False,
            )[0]
        if torch.isnan(Jt_v).any() or torch.isinf(Jt_v).any():
            Jt_v = torch.where(torch.isnan(Jt_v) | torch.isinf(Jt_v), torch.zeros_like(Jt_v), Jt_v)
        return Jt_v
    
    def backward_map(v):
        return grad_output + vjp_Jk(v)
    
    v = grad_output.clone()
    X_hist = []
    F_hist = []
    
    for i in range(max_iter):
        v_next = backward_map(v)
        if torch.isnan(v_next).any() or torch.isinf(v_next).any():
            v_next = torch.where(torch.isnan(v_next) | torch.isinf(v_next), v, v_next)
        
        X_hist.append(v.reshape(B, -1))
        F_hist.append(v_next.reshape(B, -1))
        
        if len(X_hist) < 2:
            v = v_next
            continue
        
        if len(X_hist) > m:
            X_hist = X_hist[-m:]
            F_hist = F_hist[-m:]
        
        n = len(X_hist)
        R = torch.stack([F_hist[j] - X_hist[j] for j in range(n)], dim=1)
        G = torch.bmm(R, R.transpose(1, 2))
        reg = 1e-3 if G.dtype in (torch.float16, torch.bfloat16) else 1e-6
        G = G + reg * torch.eye(n, device=v.device).unsqueeze(0)
        
        ones = torch.ones(B, n, 1, device=v.device, dtype=v.dtype)
        try:
            alpha = torch.linalg.solve(G, ones)
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
            alpha = alpha.squeeze(-1)
            if torch.isnan(alpha).any() or torch.isinf(alpha).any() or alpha.abs().max() > 1e3:
                v = v_next
                continue
        except Exception:
            v = v_next
            continue
        
        F_stack = torch.stack(F_hist, dim=1)
        v_mixed = (alpha.unsqueeze(-1) * F_stack).sum(dim=1)
        if torch.isnan(v_mixed).any() or torch.isinf(v_mixed).any():
            v = v_next
            continue
        
        rel = (v_mixed - v.reshape(B, -1)).norm() / (v.reshape(B, -1).norm() + 1e-8)
        v = v_mixed.reshape_as(v_next)
        
        if rel < tol:
            break
    
    return v


# =============================================================================
# Custom autograd Function
# =============================================================================

class FixedPointSolve(torch.autograd.Function):
    """
    Custom autograd function implementing implicit differentiation
    through a fixed point.
    """
    
    @staticmethod
    def forward(
        ctx,
        k_init: torch.Tensor,
        stats: torch.Tensor,
        step_module: nn.Module,
        max_iter_fwd: int,
        tol_fwd: float,
        anderson_m_fwd: int,
        max_iter_bwd: int,
        tol_bwd: float,
        anderson_m_bwd: int,
        *step_params,
    ):
        ctx.step_module = step_module
        ctx.max_iter_bwd = max_iter_bwd
        ctx.tol_bwd = tol_bwd
        ctx.anderson_m_bwd = anderson_m_bwd
        ctx.num_step_params = len(step_params)
        ctx.param_names = [name for name, _ in step_module.named_parameters()]
        
        with torch.no_grad():
            k = k_init.clone()
            
            if anderson_m_fwd > 0:
                k_star = _anderson_forward(
                    step_module, k, stats,
                    max_iter_fwd, tol_fwd, anderson_m_fwd
                )
            else:
                step_module.last_fwd_iters = max_iter_fwd
                step_module.last_fwd_rel = 0.0
                for i in range(max_iter_fwd):
                    k_new = step_module(k, stats, k_init)
                    if torch.isnan(k_new).any() or torch.isinf(k_new).any():
                        k_new = torch.where(torch.isnan(k_new) | torch.isinf(k_new), k_init, k_new)
                    if i > 0:
                        rel = (k_new - k).norm() / (k.norm() + 1e-8)
                        step_module.last_fwd_iters = i + 1
                        rel_val = rel.item()
                        if math.isnan(rel_val) or math.isinf(rel_val):
                            rel_val = 0.0
                        step_module.last_fwd_rel = rel_val
                        if rel < tol_fwd:
                            break
                    k = k_new
                k_star = k
                step_module.last_fwd_delta = (
                    (k_star - k_init).norm() / (k_init.norm() + 1e-8)
                ).item()
        
        ctx.save_for_backward(k_star.detach(), stats.detach(), k_init.detach())
        return k_star
    
    @staticmethod
    def backward(ctx, grad_output):
        k_star, stats, k_init = ctx.saved_tensors
        step_module = ctx.step_module
        max_iter = ctx.max_iter_bwd
        tol = ctx.tol_bwd
        anderson_m = ctx.anderson_m_bwd
        
        def g_fn_for_linear_system(k, s):
            return step_module(k, s, k_init)
        
        if anderson_m > 0:
            v = solve_fixed_point_backward_anderson(
                g_fn_for_linear_system, k_star, stats, grad_output,
                max_iter, tol, anderson_m,
            )
        else:
            v = solve_fixed_point_backward_neumann(
                g_fn_for_linear_system, k_star, stats, grad_output,
                max_iter, tol,
            )
        
        with torch.enable_grad():
            k_in = k_star.detach().requires_grad_(True)
            s_in = stats.detach().requires_grad_(True)
            base_in = k_init.detach().requires_grad_(True)
            
            for p in step_module.parameters():
                p.requires_grad_(True)
            
            g_out = step_module(k_in, s_in, base_in)
            
            all_inputs = [base_in, s_in] + list(step_module.parameters())
            all_grads = torch.autograd.grad(
                outputs=g_out,
                inputs=all_inputs,
                grad_outputs=v,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
        
        grad_k_init = all_grads[0]
        grad_stats = all_grads[1]
        grad_params = all_grads[2:]
        
        if grad_k_init is not None and (torch.isnan(grad_k_init).any() or torch.isinf(grad_k_init).any()):
            grad_k_init = torch.where(torch.isnan(grad_k_init) | torch.isinf(grad_k_init), torch.zeros_like(grad_k_init), grad_k_init)
            
        if grad_stats is not None and (torch.isnan(grad_stats).any() or torch.isinf(grad_stats).any()):
            grad_stats = torch.where(torch.isnan(grad_stats) | torch.isinf(grad_stats), torch.zeros_like(grad_stats), grad_stats)
        
        grad_step_params = tuple(
            torch.where(torch.isnan(g) | torch.isinf(g), torch.zeros_like(g), g) if g is not None else torch.zeros_like(p)
            for g, p in zip(grad_params, step_module.parameters())
        )
        
        return (
            grad_k_init,
            grad_stats,
            None,
            None, None, None,
            None, None, None,
            *grad_step_params,
        )


# =============================================================================
# Anderson acceleration for forward pass
# =============================================================================

def _anderson_forward(step_module, k_init, stats, max_iter, tol, m):
    """Anderson-accelerated forward fixed-point iteration."""
    B = k_init.shape[0]
    k = k_init.clone()
    X_hist, F_hist = [], []
    
    # Defaults in case of early exit or failure
    step_module.last_fwd_iters = max_iter
    step_module.last_fwd_rel = 0.0
    step_module.last_fwd_alpha = step_module.alpha.detach().item()
    
    for i in range(max_iter):
        k_next = step_module(k, stats, k_init)
        if torch.isnan(k_next).any() or torch.isinf(k_next).any():
            k_next = torch.where(torch.isnan(k_next) | torch.isinf(k_next), k, k_next)

        rel = (k_next - k).norm() / (k.norm() + 1e-8)
        step_module.last_fwd_iters = i + 1
        rel_val = rel.item()
        if math.isnan(rel_val) or math.isinf(rel_val):
            rel_val = 0.0
        step_module.last_fwd_rel = rel_val
        if rel < tol:
            step_module.last_fwd_delta = (
                (k_next - k_init).norm() / (k_init.norm() + 1e-8)
            ).item()
            return k_next
        
        X_hist.append(k.reshape(B, -1))
        F_hist.append(k_next.reshape(B, -1))
        
        if len(X_hist) < 2:
            k = k_next
            continue
        
        if len(X_hist) > m:
            X_hist = X_hist[-m:]
            F_hist = F_hist[-m:]
        
        n = len(X_hist)
        R = torch.stack([F_hist[j] - X_hist[j] for j in range(n)], dim=1)
        G = torch.bmm(R, R.transpose(1, 2))
        reg = 1e-3 if G.dtype in (torch.float16, torch.bfloat16) else 1e-6
        G = G + reg * torch.eye(n, device=k.device).unsqueeze(0)
        
        ones = torch.ones(B, n, 1, device=k.device, dtype=k.dtype)
        try:
            alpha = torch.linalg.solve(G, ones)
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
            alpha = alpha.squeeze(-1)
            if torch.isnan(alpha).any() or torch.isinf(alpha).any() or alpha.abs().max() > 1e3:
                k = k_next
                continue
        except Exception:
            k = k_next
            continue
        
        F_stack = torch.stack(F_hist, dim=1)
        k_mixed = (alpha.unsqueeze(-1) * F_stack).sum(dim=1)
        if torch.isnan(k_mixed).any() or torch.isinf(k_mixed).any():
            k = k_next
            continue
        
        rel = (k_mixed - k.reshape(B, -1)).norm() / (k.reshape(B, -1).norm() + 1e-8)
        k = k_mixed.reshape_as(k_next)
        
        step_module.last_fwd_iters = i + 1
        rel_val = rel.item()
        if math.isnan(rel_val) or math.isinf(rel_val):
            rel_val = 0.0
        step_module.last_fwd_rel = rel_val
        
        if rel < tol:
            break

    step_module.last_fwd_delta = (
        (k - k_init).norm() / (k_init.norm() + 1e-8)
    ).item()
    return k
