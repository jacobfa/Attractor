import torch


def _rel_residual(y_new: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
    yn = y_new.float()
    yp = y_prev.float()
    diff = (yn - yp).reshape(yn.size(0), -1).norm(dim=1)
    ref = yp.reshape(yp.size(0), -1).norm(dim=1).clamp_min(1e-9)
    return diff / ref


def fpi_solve(f, y0, max_iter: int = 30, tol: float = 1e-3,
              min_iter: int = 0, return_trajectory: bool = False):
    y = y0
    trajectory = [y0] if return_trajectory else None
    rel = torch.zeros(y0.size(0), device=y0.device)
    iters = 0
    converged = False
    min_iter = max(int(min_iter), 0)
    for k in range(max_iter):
        y_new = f(y)
        if return_trajectory:
            trajectory.append(y_new)
        rel = _rel_residual(y_new, y)
        iters = k + 1
        y = y_new
        if iters >= min_iter and rel.max().item() < tol:
            converged = True
            break
    return y, {
        "iters": iters,
        "rel_residual": rel.max().item(),
        "converged": converged,
        "trajectory": trajectory,
    }


def anderson_solve(f, y0, max_iter: int = 30, tol: float = 1e-3,
                   m: int = 5, beta: float = 1.0, lam: float = 1e-4,
                   min_iter: int = 0, return_trajectory: bool = False):
    # Batched type-II Anderson acceleration for y = f(y).
    B = y0.size(0)
    shape = y0.shape
    n = int(y0.numel() // B)
    dev, dt = y0.device, y0.dtype

    X = torch.zeros(B, m, n, device=dev, dtype=dt)
    Fh = torch.zeros(B, m, n, device=dev, dtype=dt)
    X[:, 0] = y0.reshape(B, n)
    Fh[:, 0] = f(y0).reshape(B, n)
    X[:, 1] = Fh[:, 0]
    Fh[:, 1] = f(Fh[:, 0].view(shape)).reshape(B, n)

    H = torch.zeros(B, m + 1, m + 1, device=dev, dtype=dt)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    rhs = torch.zeros(B, m + 1, 1, device=dev, dtype=dt)
    rhs[:, 0] = 1.0

    trajectory = None
    if return_trajectory:
        trajectory = [y0, Fh[:, 0].view(shape), Fh[:, 1].view(shape)]

    rel_max = 0.0
    converged = False
    last_idx = 1
    iters = 2
    min_iter = max(int(min_iter), 0)
    for k in range(2, max_iter):
        n_act = min(k, m)
        G = Fh[:, :n_act] - X[:, :n_act]
        H[:, 1:n_act + 1, 1:n_act + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n_act, device=dev, dtype=dt)[None]
        )
        H_fp = H[:, :n_act + 1, :n_act + 1].float()
        rhs_fp = rhs[:, :n_act + 1].float()
        sol, info = torch.linalg.solve_ex(H_fp, rhs_fp)
        if (info != 0).any():
            try:
                lstsq_sol = torch.linalg.lstsq(H_fp, rhs_fp).solution
                bad = (info != 0).view(-1, 1, 1).expand_as(sol)
                sol = torch.where(bad, lstsq_sol, sol)
            except Exception:
                pass
        sol = sol.to(dt)
        alpha = sol[:, 1:n_act + 1, 0]
        if not torch.isfinite(alpha).all():
            alpha_fpi = torch.zeros_like(alpha)
            alpha_fpi[:, -1] = 1.0
            alpha = torch.where(
                torch.isfinite(alpha).all(dim=-1, keepdim=True),
                alpha, alpha_fpi)
        new_X = (
            beta * (alpha[..., None] * Fh[:, :n_act]).sum(dim=1)
            + (1.0 - beta) * (alpha[..., None] * X[:, :n_act]).sum(dim=1)
        )
        if not torch.isfinite(new_X).all():
            new_X = Fh[:, (k - 1) % m].clone()
        idx = k % m
        X[:, idx] = new_X
        Fh[:, idx] = f(new_X.view(shape)).reshape(B, n)
        last_idx = idx
        iters = k + 1
        if return_trajectory:
            trajectory.append(Fh[:, idx].view(shape))

        rel = _rel_residual(Fh[:, idx].view(shape), X[:, idx].view(shape))
        rel_max = rel.max().item()
        if iters >= min_iter and rel_max < tol:
            converged = True
            break
        if (rel_max != rel_max) or rel_max > 1e4:
            break

    y_star = Fh[:, last_idx].view(shape)
    return y_star, {
        "iters": iters,
        "rel_residual": rel_max,
        "converged": converged,
        "trajectory": trajectory,
    }


SOLVERS = {"anderson": anderson_solve, "fpi": fpi_solve}


def get_solver(name: str):
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver '{name}'. Choices: {list(SOLVERS)}")
    return SOLVERS[name]
