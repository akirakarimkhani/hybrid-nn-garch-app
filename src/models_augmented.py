"""
Augmented hybrid GARCH-NN models ported from 02_augmented_architecture.ipynb.
Three variants:
  1. Donaldson-Kamstra (PyTorch, sigmoid basis)
  2. Extended-MLP (PyTorch, joint GARCH + MLP)
  3. Extended-LSTM (TensorFlow, joint GARCH + LSTM)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import forecast_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Donaldson-Kamstra Approach
# ═══════════════════════════════════════════════════════════════════════════════

class AugmentedHybridModel(nn.Module):
    def __init__(self, p=1, q=1, s=10, d=1, m=2):
        super().__init__()
        self.p, self.q, self.s, self.d, self.m = p, q, s, d, m

        self.omega = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor([0.1] * q))
        self.beta = nn.Parameter(torch.tensor([0.8] * p))
        self.zeta = nn.Parameter(torch.randn(s) * 0.01)
        self.lambda_hdw = nn.Parameter(
            0.5 * torch.empty((s, d, m)).uniform_(-1, 1), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(s))

    def forward(self, eps, sigma_sq):
        T = len(eps)
        vals = []
        eps_mean = eps.mean()
        eps_std = eps.std()
        eps_std = eps_std if eps_std > 1e-6 else torch.tensor(1.0)

        for t in range(T):
            garch_sum = torch.sum(self.alpha * eps[max(0, t - self.q):t] ** 2) if t > 0 else 0.0
            beta_sum = torch.sum(self.beta * sigma_sq[max(0, t - self.p):t]) if t > 0 else 0.0
            garch_part = self.omega + garch_sum + beta_sum

            z_t = [
                (eps[t - dd] - eps_mean) / eps_std if (t - dd) >= 0 else 0.0
                for dd in range(1, self.d + 1)
            ]
            z_t = torch.tensor(z_t)

            basis = []
            for h_idx in range(self.s):
                lin_combo = sum(
                    self.lambda_hdw[h_idx, dd, ww] * z_t[dd]
                    for dd in range(self.d)
                    for ww in range(self.m)
                )
                basis.append(torch.sigmoid(lin_combo))
            basis = torch.stack(basis)
            correction = torch.sum(self.zeta * basis)

            val = torch.clamp(garch_part + correction, min=1e-4)
            vals.append(val)

        return torch.stack(vals)


def _nll_torch(y_true, y_pred):
    y_pred = torch.clamp(y_pred, min=1e-6)
    return 0.5 * torch.mean(torch.log(y_pred) + y_true / y_pred)


def train_aug_donaldson(train_returns: pd.Series, epochs: int = 100,
                        progress_cb=None):
    """
    Train AugmentedHybridModel (Donaldson) on train_returns.
    Returns (model, loss_history, params_dict).
    """
    X = torch.tensor(train_returns.values.flatten()).float()
    y = X ** 2

    model = AugmentedHybridModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        h = model(X, X ** 2)
        loss = _nll_torch(y, h)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    params = {
        "Omega": float(model.omega.detach().numpy()),
        "Alpha": float(model.alpha.detach().numpy().mean()),
        "Beta": float(model.beta.detach().numpy().mean()),
        "NN Adjust (ζ mean)": float(model.zeta.detach().numpy().mean()),
    }
    return model, losses, params


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Extended Approach — MLP (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class AugmentedModel_Extended_mlp(nn.Module):
    def __init__(self, input_size=2, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return torch.tanh(self.fc2(F.relu(self.fc1(x))))


class ConstrainedGARCHParams(nn.Module):
    def __init__(self, init_omega=0.1, init_alpha=0.1, init_beta=0.8):
        super().__init__()
        self.raw_omega = nn.Parameter(torch.tensor([init_omega]).log())
        self.raw_alpha = nn.Parameter(torch.tensor(init_alpha).logit())
        self.raw_beta = nn.Parameter(torch.tensor(init_beta).logit())

    def forward(self):
        omega = F.softplus(self.raw_omega)
        alpha = torch.sigmoid(self.raw_alpha)
        beta = torch.sigmoid(self.raw_beta) * (1 - alpha)
        return omega.squeeze(), alpha.squeeze(), beta.squeeze()


def _compute_volatility_joint(returns_tensor, net, garch_param_module):
    T = len(returns_tensor)
    h_list = []
    h_prev = torch.var(returns_tensor)
    omega, alpha, beta = garch_param_module()
    mean_r2 = returns_tensor.pow(2).mean().item()
    mean_h = torch.var(returns_tensor).item()
    nn_adjust = torch.tensor(0.0)

    for t in range(T):
        if t == 0:
            h_t = h_prev
        else:
            r2 = returns_tensor[t - 1] ** 2
            h_det = h_prev.detach()
            nn_in = torch.stack([r2 / mean_r2, h_det / mean_h]).unsqueeze(0)
            nn_adjust = net(nn_in).squeeze()
            h_t = torch.clamp(omega + alpha * r2 + beta * h_prev + nn_adjust, min=1e-4)
        h_list.append(h_t.view(()))
        h_prev = h_t

    return torch.stack(h_list), nn_adjust


def _nll_joint(returns_tensor, h):
    h = torch.clamp(h, min=1e-6)
    eps = returns_tensor / torch.sqrt(h)
    return 0.5 * (torch.log(h) + eps ** 2).mean()


def run_joint_training(aug_model, train_returns: pd.Series, epochs: int = 100,
                       init_omega=0.1, init_alpha=0.1, init_beta=0.8,
                       progress_cb=None):
    """
    Joint training of GARCH parameters + MLP adjustment.
    Returns (net, loss_history, params_dict, garch_param_module).
    """
    returns_tensor = torch.tensor(train_returns.values.flatten()).float()
    net = aug_model
    gp = ConstrainedGARCHParams(init_omega, init_alpha, init_beta)
    optimizer = optim.Adam(list(net.parameters()) + list(gp.parameters()), lr=0.001)
    losses = []

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        h, nn_adj = _compute_volatility_joint(returns_tensor, net, gp)
        loss = _nll_joint(returns_tensor, h)
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    omega, alpha, beta = gp()
    params = {
        "Omega": float(omega.item()),
        "Alpha": float(alpha.item()),
        "Beta": float(beta.item()),
        "NN Adjust": float(nn_adj.item() if hasattr(nn_adj, 'item') else nn_adj),
    }
    return net, losses, params, gp


@torch.no_grad()
def aug_mlp_multistep_preds(net, garch_param_module, test_returns: pd.Series,
                            H: int = 100, min_var: float = 1e-6,
                            lower_triangular: bool = False,
                            progress_cb=None):
    """
    Build lower-triangular forecast matrix for the Aug-MLP model.
    Returns (preds_df, qlike_array).
    """
    device = next(net.parameters()).device
    net.eval()

    r_full = torch.tensor(test_returns.values, dtype=torch.float32, device=device).view(-1, 1)
    N = r_full.shape[0]
    H = min(H, N)

    mean_r2 = max(float(r_full.pow(2).mean()), 1e-6)
    mean_h = max(float(r_full.var()), 1e-6)

    omega, alpha, beta = garch_param_module()
    omega, alpha, beta = (x.to(device).view(1, 1) for x in (omega, alpha, beta))

    h_prev = r_full.var()
    h0_list = []
    for t in range(N):
        if t == 0:
            h_t = h_prev
        else:
            r2 = r_full[t - 1, 0] ** 2
            nn_in = torch.stack([r2 / mean_r2, h_prev.squeeze() / mean_h]).view(1, 2)
            nn_adj = net(nn_in).view(1, 1)
            h_t = torch.clamp(omega + alpha * r2 + beta * h_prev + nn_adj, min=min_var)
        h0_list.append(h_t.view(1, 1))
        h_prev = h_t
    h_start = torch.cat(h0_list, dim=0)

    mat = np.full((N, H), np.nan, dtype=np.float64)
    r2_curr = r_full.pow(2)
    h_curr = h_start.clone()

    for h in range(1, H + 1):
        count = (N - h) if lower_triangular else N
        if count <= 0:
            break
        feats = torch.cat([r2_curr[:count] / mean_r2,
                           h_curr[:count].detach() / mean_h], dim=1)
        nn_adj = net(feats).view(-1, 1)
        v_hat = torch.clamp(omega + alpha * r2_curr[:count] + beta * h_curr[:count] + nn_adj,
                            min=min_var)
        mat[:count, h - 1] = v_hat.squeeze(-1).cpu().numpy()
        r2_curr = torch.cat([v_hat, r2_curr[count:]], dim=0)
        h_curr = torch.cat([v_hat, h_curr[count:]], dim=0)
        if progress_cb:
            progress_cb(h, H)

    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    preds_df = pd.DataFrame(mat, index=test_returns.index, columns=cols)

    qlike_arr = []
    proxy = (test_returns ** 2).values
    for i in range(H):
        p = preds_df.iloc[:, i].dropna().values
        y = proxy[:len(p)]
        mask = np.isfinite(p) & (p > 0) & np.isfinite(y) & (y > 0)
        if mask.sum() == 0:
            qlike_arr.append(np.nan)
        else:
            qlike_arr.append(float(np.mean(np.log(p[mask]) + y[mask] / p[mask])))

    return preds_df, np.array(qlike_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Extended Approach — LSTM (TensorFlow)
# ═══════════════════════════════════════════════════════════════════════════════

import tensorflow as tf


class ConstrainedGARCHParamsTF(tf.keras.Model):
    def __init__(self, init_omega=0.1, init_alpha=0.1, init_beta=0.8):
        super().__init__()
        self.raw_omega = tf.Variable(
            tf.math.log(tf.constant([init_omega], dtype=tf.float32)), trainable=True
        )
        self.raw_alpha = tf.Variable(
            tf.math.log(init_alpha / (1 - init_alpha)), trainable=True
        )
        self.raw_beta = tf.Variable(
            tf.math.log(init_beta / (1 - init_beta)), trainable=True
        )

    def __call__(self):
        omega = tf.nn.softplus(self.raw_omega)
        alpha = tf.nn.sigmoid(self.raw_alpha)
        beta = tf.nn.sigmoid(self.raw_beta) * (1 - alpha)
        return tf.squeeze(omega), tf.squeeze(alpha), tf.squeeze(beta)


def build_lstm_adjustment_model(input_dim=2, lstm_units=16):
    from tensorflow.keras.layers import Input, LSTM, Dense
    inputs = tf.keras.Input(shape=(1, input_dim))
    x = LSTM(lstm_units, return_sequences=False)(inputs)
    x = Dense(8, activation='relu')(x)
    out = Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


@tf.function()
def _compute_vol_lstm_fast(returns_tensor, net, garch_params,
                           r2_mean, r2_std, h_mean, h_std,
                           min_var=1e-4, max_var=1e4):
    x = tf.cast(returns_tensor, tf.float32)
    T = tf.shape(x)[0]
    r2_series = tf.square(x)
    tiny = tf.constant(1e-8, tf.float32)
    r2_std = tf.maximum(r2_std, tiny)
    h_std = tf.maximum(h_std, tiny)

    omega, alpha, beta = garch_params()
    h_ta = tf.TensorArray(tf.float32, size=T)
    h_prev = tf.math.reduce_variance(x)
    nn_last = tf.constant(0.0, tf.float32)
    h_ta = h_ta.write(0, tf.clip_by_value(h_prev, min_var, max_var))

    def body(t, h_prev, nn_last, h_ta):
        r2 = r2_series[t - 1]
        h_det = tf.stop_gradient(h_prev)
        x1 = (r2 - r2_mean) / r2_std
        x2 = (h_det - h_mean) / h_std
        nn_in = tf.reshape(tf.stack([x1, x2]), [1, 1, 2])
        nn_adj = net(nn_in)[:, 0][0]
        h_t = tf.clip_by_value(omega + alpha * r2 + beta * h_prev + nn_adj,
                               min_var, max_var)
        h_ta = h_ta.write(t, h_t)
        return t + 1, h_t, nn_adj, h_ta

    _, _, nn_last, h_ta = tf.while_loop(
        lambda t, *_: t < T, body,
        (tf.constant(1), h_prev, nn_last, h_ta),
        parallel_iterations=1,
    )
    return h_ta.stack(), nn_last


def _nll_tf(returns_tensor, h):
    h = tf.clip_by_value(h, 1e-6, 1e4)
    eps = returns_tensor / tf.sqrt(h)
    return tf.reduce_mean(0.5 * (tf.math.log(h) + tf.square(eps)))


def run_lstm_joint_training(train_returns: pd.Series, epochs: int = 100,
                            init_omega=0.1, init_alpha=0.1, init_beta=0.8,
                            progress_cb=None):
    """
    Joint training of GARCH parameters + LSTM adjustment (TensorFlow).
    Returns (net, garch_params, loss_history, params_dict).
    """
    rt = tf.convert_to_tensor(train_returns.values.flatten(), dtype=tf.float32)
    net = build_lstm_adjustment_model()
    gp = ConstrainedGARCHParamsTF(init_omega, init_alpha, init_beta)
    gp.build(input_shape=())

    r2 = tf.square(rt)
    r2_mean = tf.reduce_mean(r2)
    r2_std = tf.math.reduce_std(r2)
    h_var = tf.math.reduce_variance(rt)
    h_mean = h_var
    h_std = tf.sqrt(h_mean)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    losses = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            h, nn_adj = _compute_vol_lstm_fast(rt, net, gp, r2_mean, r2_std, h_mean, h_std)
            loss = _nll_tf(rt, h)
        grads = tape.gradient(loss, net.trainable_variables + gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables + gp.trainable_variables))
        losses.append(float(loss.numpy()))
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    omega, alpha, beta = gp()
    params = {
        "Omega": float(omega.numpy()),
        "Alpha": float(alpha.numpy()),
        "Beta": float(beta.numpy()),
        "NN Adjust": float(nn_adj.numpy() if hasattr(nn_adj, 'numpy') else nn_adj),
    }
    return net, gp, losses, params


def aug_lstm_multistep_preds(net, garch_params, test_returns: pd.Series,
                             H: int = 100, min_var: float = 1e-6,
                             lower_triangular: bool = False,
                             progress_cb=None):
    """
    Build forecast matrix for the Aug-LSTM model.
    Returns (preds_df, qlike_array).
    """
    r_full = tf.convert_to_tensor(test_returns.values.astype("float32"))
    N = int(r_full.shape[0])
    H = min(H, N)

    mean_r2 = tf.maximum(tf.reduce_mean(tf.square(r_full)), 1e-6)
    mean_h = tf.maximum(tf.math.reduce_variance(r_full), 1e-6)
    omega, alpha, beta = garch_params()
    omega = tf.cast(omega, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    beta = tf.cast(beta, tf.float32)

    h_prev = tf.math.reduce_variance(r_full)
    h_list = []
    for t in range(N):
        if t == 0:
            h_t = h_prev
        else:
            r2_tm1 = tf.square(r_full[t - 1])
            h_det = tf.stop_gradient(h_prev)
            feats = tf.stack([r2_tm1 / mean_r2, h_det / mean_h])
            feats = tf.expand_dims(tf.expand_dims(feats, 0), 1)
            nn_adj = net(feats)[0, 0]
            h_t = tf.clip_by_value(
                omega + alpha * r2_tm1 + beta * h_prev + nn_adj, min_var, 1e12
            )
        h_list.append(tf.reshape(h_t, (1, 1)))
        h_prev = h_t
    h_start = tf.concat(h_list, axis=0)

    mat = np.full((N, H), np.nan, dtype=np.float64)
    r2_curr = tf.square(tf.reshape(r_full, (-1, 1)))
    h_curr = tf.identity(h_start)

    for h in range(1, H + 1):
        count = (N - h) if lower_triangular else N
        if count <= 0:
            break
        r2_s = r2_curr[:count] / mean_r2
        h_s = tf.stop_gradient(h_curr[:count]) / mean_h
        feats = tf.concat([r2_s, h_s], axis=1)
        feats = tf.expand_dims(feats, 1)
        nn_adj = tf.reshape(net(feats), (-1, 1))
        v_hat = tf.clip_by_value(
            omega + alpha * r2_curr[:count] + beta * h_curr[:count] + nn_adj,
            min_var, 1e12
        )
        mat[:count, h - 1] = v_hat.numpy().reshape(-1)
        r2_curr = tf.concat([v_hat, r2_curr[count:]], axis=0)
        h_curr = tf.concat([v_hat, h_curr[count:]], axis=0)
        if progress_cb:
            progress_cb(h, H)

    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    preds_df = pd.DataFrame(mat, index=test_returns.index, columns=cols)

    qlike_arr = []
    proxy = (test_returns ** 2).values
    for i in range(H):
        p = preds_df.iloc[:, i].dropna().values
        y = proxy[:len(p)]
        mask = np.isfinite(p) & (p > 0) & np.isfinite(y) & (y > 0)
        if mask.sum() == 0:
            qlike_arr.append(np.nan)
        else:
            qlike_arr.append(float(np.mean(np.log(p[mask]) + y[mask] / p[mask])))

    return preds_df, np.array(qlike_arr)
