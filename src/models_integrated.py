"""
Integrated hybrid GARCH-NN models ported from 03_integrated_architecture.ipynb.
Two variants: IntegratedMLP (pure GARCH, learnable params) and IntegratedLSTM.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1 Integrated-MLP (GARCH with learnable parameters, no explicit NN layer)
# ═══════════════════════════════════════════════════════════════════════════════

class IntegratedMLP(nn.Module):
    def __init__(self, alpha_init=0.1, beta_init=0.85, omega_init=0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.beta = nn.Parameter(torch.tensor([beta_init]))
        self.omega_raw = nn.Parameter(torch.tensor([omega_init]).log().exp())

    def forward(self, eps_sq_tm1, sigma_sq_tm1):
        omega = F.softplus(self.omega_raw)
        sigma_sq = omega + self.alpha * eps_sq_tm1 + self.beta * sigma_sq_tm1
        return F.softplus(sigma_sq)


def _nll_integ(y, sigma_sq):
    return 0.5 * torch.mean(torch.log(sigma_sq) + y ** 2 / sigma_sq)


@torch.no_grad()
def _filter_sigma2(series_norm: torch.Tensor, model: nn.Module,
                   initial_var: torch.Tensor) -> torch.Tensor:
    T = len(series_norm)
    out = []
    sigma_prev = initial_var
    for t in range(T):
        if t == 0:
            out.append(sigma_prev)
        else:
            eps_sq = series_norm[t - 1].view(1, 1) ** 2
            s_tm1 = sigma_prev.view(1, 1)
            s_t = model(eps_sq, s_tm1).squeeze()
            out.append(s_t)
            sigma_prev = s_t
    return torch.stack(out)


def train_integrated_mlp(returns: pd.Series, train_ratio: float = 0.8,
                         epochs: int = 100, lr: float = 0.001,
                         warmup_window: int = 20, progress_cb=None):
    """
    Train IntegratedMLP on the full returns tensor.
    Returns (model, sigma_sq_test, loss_history, params_dict).
    """
    returns_tensor = torch.tensor(returns.values, dtype=torch.float32)
    T = len(returns_tensor)
    T_train = int(T * train_ratio)

    mean = returns_tensor[:T_train].mean()
    std = returns_tensor[:T_train].std()
    returns_norm = (returns_tensor - mean) / std
    train_norm = returns_norm[:T_train]
    test_norm = returns_norm[T_train:]

    initial_var = torch.var(train_norm[:warmup_window])
    omega_guess = initial_var * (1 - 0.1 - 0.85)
    model = IntegratedMLP(omega_init=float(omega_guess))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        sigma_list = []
        s_prev = initial_var
        for t in range(T_train):
            if t == 0:
                sigma_list.append(s_prev)
                continue
            eps_sq = train_norm[t - 1].view(1, 1) ** 2
            s_tm1 = s_prev.view(1, 1)
            s_t = model(eps_sq, s_tm1).squeeze()
            sigma_list.append(s_t)
            s_prev = s_t

        sigma_train = torch.stack(sigma_list)
        loss = _nll_integ(train_norm[warmup_window:], sigma_train[warmup_window:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    # generate test sigma
    with torch.no_grad():
        sigma_all = _filter_sigma2(returns_norm, model, initial_var)
        sigma_test = sigma_all[T_train:] * (std ** 2)

    omega_v = float(F.softplus(model.omega_raw).item())
    params = {
        "Omega": omega_v,
        "Alpha": float(model.alpha.item()),
        "Beta": float(model.beta.item()),
    }
    return model, sigma_test, losses, params, returns_tensor


@torch.no_grad()
def integrated_mlp_multistep_preds(model: IntegratedMLP,
                                   returns_tensor: torch.Tensor,
                                   train_ratio: float = 0.8,
                                   H: int = 100,
                                   warmup_window: int = 20,
                                   test_index=None,
                                   progress_cb=None):
    """
    Lower-triangular multi-step forecast for IntegratedMLP.
    Returns (preds_df, qlike_array).
    """
    T = len(returns_tensor)
    T_train = int(T * train_ratio)

    mean = returns_tensor[:T_train].mean()
    std = returns_tensor[:T_train].std()
    series_norm = (returns_tensor - mean) / std

    initial_var = torch.var(series_norm[:warmup_window])
    sigma_all = _filter_sigma2(series_norm, model, initial_var)

    test_norm = series_norm[T_train:]
    origins = len(test_norm)
    F_mat = torch.zeros((origins, H), dtype=series_norm.dtype)

    for i in range(origins):
        t_abs = T_train + i
        eps_sq_prev = series_norm[t_abs] ** 2
        sigma_prev = sigma_all[t_abs]
        for h in range(H):
            sigma_h = model(eps_sq_prev.view(1, 1), sigma_prev.view(1, 1)).squeeze()
            F_mat[i, h] = sigma_h
            eps_sq_prev = sigma_h
            sigma_prev = sigma_h
        if progress_cb:
            progress_cb(i + 1, origins)

    F_rescaled = (F_mat * (std ** 2)).numpy()
    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    idx = test_index if test_index is not None else range(origins)
    preds_df = pd.DataFrame(F_rescaled, index=idx, columns=cols)

    test_sq = (returns_tensor[T_train:] ** 1).numpy()  # using unnormalized returns
    test_sq_orig = ((returns_tensor[T_train:]) ** 2).numpy()
    qlike_arr = []
    for i in range(H):
        p = preds_df.iloc[:, i].values
        y = test_sq_orig
        n = min(len(p), len(y))
        p, y = p[:n], y[:n]
        mask = np.isfinite(p) & (p > 0) & np.isfinite(y) & (y > 0)
        if mask.sum() == 0:
            qlike_arr.append(np.nan)
        else:
            qlike_arr.append(float(np.mean(np.log(p[mask]) + y[mask] / p[mask])))

    return preds_df, np.array(qlike_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.2 Integrated-LSTM
# ═══════════════════════════════════════════════════════════════════════════════

class IntegratedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(1, hidden_size, bias=False)
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(1, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(1, hidden_size, bias=False)
        self.w = nn.Parameter(torch.tensor([0.1]))
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.beta = nn.Parameter(torch.tensor([0.85]))
        self.omega_raw = nn.Parameter(torch.tensor([0.01]).log().exp())

    def forward(self, eps_sq_tm1, sigma_sq_tm1, c_tm1):
        f_t = torch.sigmoid(self.W_f(eps_sq_tm1) + self.U_f(sigma_sq_tm1))
        i_t = torch.sigmoid(self.W_i(eps_sq_tm1) + self.U_i(sigma_sq_tm1))
        c_tilde = torch.tanh(self.W_c(eps_sq_tm1) + self.U_c(sigma_sq_tm1))
        c_t = f_t * c_tm1 + i_t * c_tilde
        omega = F.softplus(self.omega_raw)
        o_t = omega + self.alpha * eps_sq_tm1 + self.beta * sigma_sq_tm1
        sigma_sq_t = o_t * (1 + self.w * torch.tanh(c_t).mean(dim=1, keepdim=True))
        return sigma_sq_t, c_t


def train_integrated_lstm(train_returns: pd.Series, epochs: int = 100,
                          lr: float = 0.01, progress_cb=None):
    """
    Train IntegratedLSTM on train_returns.
    Returns (model, sigma_sq_init, c_init, loss_history, params_dict).
    """
    sigma_sq_init = torch.tensor(
        float((train_returns.iloc[:2] ** 2).mean())
    ).unsqueeze(0).unsqueeze(1).float()
    sq_returns = torch.tensor(train_returns.values[:-1] ** 2).unsqueeze(1)
    ret_t = torch.tensor(train_returns.values[1:]).unsqueeze(1).float()
    c_init = torch.zeros(1, 16)

    model = IntegratedLSTM()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        s_tm1 = sigma_sq_init.clone()
        c_t = c_init.clone()
        total_loss = torch.tensor(0.0)
        for t in range(len(sq_returns)):
            eps_sq = sq_returns[t].unsqueeze(0).float()
            r_t = ret_t[t].unsqueeze(0).float()
            s_t, c_t = model(eps_sq, s_tm1, c_t)
            nll = 0.5 * (torch.log(s_t) + r_t ** 2 / s_t)
            total_loss = total_loss + nll.mean()
            s_tm1 = s_t.detach()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    omega_v = float(F.softplus(model.omega_raw).item())
    params = {
        "Omega": omega_v,
        "Alpha": float(model.alpha.item()),
        "Beta": float(model.beta.item()),
        "w (cell weight)": float(model.w.item()),
    }
    return model, sigma_sq_init, c_init, losses, params


@torch.no_grad()
def integrated_lstm_multistep_preds(model: IntegratedLSTM,
                                    test_returns: pd.Series,
                                    sigma_sq_init: torch.Tensor,
                                    c_init: torch.Tensor,
                                    H: int = 100,
                                    clamp_min: float = 1e-12,
                                    lower_triangular: bool = False,
                                    progress_cb=None):
    """
    Build forecast matrix for IntegratedLSTM.
    Returns (preds_df, qlike_array).
    """
    device = next(model.parameters()).device
    x = torch.tensor(test_returns.values ** 2, dtype=torch.float32, device=device)
    sigma_prev = sigma_sq_init.to(device).float()
    c_prev = c_init.to(device).float()
    T = len(x)

    sigma_filt = torch.zeros(T, 1, device=device)
    c_states = torch.zeros(T, c_prev.shape[1], device=device)
    sigma_filt[0:1] = sigma_prev
    c_states[0:1] = c_prev

    for i in range(1, T):
        eps_sq = x[i - 1].view(1, 1)
        s_i, c_prev = model(eps_sq, sigma_prev, c_prev)
        s_i = torch.clamp(s_i, min=clamp_min)
        sigma_filt[i:i + 1] = s_i
        c_states[i:i + 1] = c_prev
        sigma_prev = s_i.detach()

    mat = np.full((T, H), np.nan, dtype=np.float64)
    for i in range(T):
        eps_sq_p = x[i].view(1, 1)
        s_p = sigma_filt[i].view(1, 1)
        c_p = c_states[i].view(1, -1)
        max_h = (T - i) if lower_triangular else H
        for h in range(min(H, max_h)):
            s_h, c_p = model(eps_sq_p, s_p, c_p)
            s_h = torch.clamp(s_h, min=clamp_min)
            mat[i, h] = float(s_h.squeeze())
            eps_sq_p = s_h
            s_p = s_h
        if progress_cb:
            progress_cb(i + 1, T)

    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    preds_df = pd.DataFrame(mat, index=test_returns.index, columns=cols)

    proxy = (test_returns ** 2).values
    qlike_arr = []
    for i in range(H):
        p = preds_df.iloc[:, i].dropna().values
        y = proxy[:len(p)]
        mask = np.isfinite(p) & (p > 0) & np.isfinite(y) & (y > 0)
        if mask.sum() == 0:
            qlike_arr.append(np.nan)
        else:
            qlike_arr.append(float(np.mean(np.log(p[mask]) + y[mask] / p[mask])))

    return preds_df, np.array(qlike_arr)
