"""
Utility functions ported from 01_utilities.ipynb.
All file I/O and global-variable dependencies have been removed.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ── Loss ─────────────────────────────────────────────────────────────────────

def gaussian_nll(y_true, y_pred):
    """Gaussian negative log-likelihood as a Keras loss."""
    import tensorflow as tf
    sigma2 = y_pred
    return 0.5 * (tf.math.log(sigma2) + tf.square(y_true) / sigma2)


# ── Data Preparation ─────────────────────────────────────────────────────────

def train_test_split_nn(return_input, split_ratio, scaling=True,
                        scaler="Standard", reshape_dim=2):
    """
    Prepare (X, y) pairs of squared returns for MLP training.
    X[t] = r²_t, y[t] = r²_{t+1}
    """
    split = int(len(return_input) * split_ratio)
    X = np.array(return_input) ** 2
    y = np.array(return_input[1:]) ** 2

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if reshape_dim == 2:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    elif reshape_dim == 3:
        X_train = X_train.reshape(-1, 1, 1)
        X_test = X_test.reshape(-1, 1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    if scaling:
        sc_x = StandardScaler() if scaler == "Standard" else MinMaxScaler()
        sc_y = StandardScaler() if scaler == "Standard" else MinMaxScaler()

        if reshape_dim == 2:
            X_train = sc_x.fit_transform(X_train)
            X_test = sc_x.transform(X_test)
            y_train = sc_y.fit_transform(y_train)
            y_test = sc_y.transform(y_test)
        elif reshape_dim == 3:
            X_train = sc_x.fit_transform(X_train.reshape(-1, 1)).reshape(-1, 1, 1)
            X_test = sc_x.transform(X_test.reshape(-1, 1)).reshape(-1, 1, 1)
            y_train = sc_y.fit_transform(y_train)
            y_test = sc_y.transform(y_test)

        return X_train, X_test, y_train, y_test, sc_x, sc_y

    return X_train, X_test, y_train, y_test


def make_windows(arr, win):
    X, y = [], []
    for i in range(len(arr) - win):
        X.append(arr[i: i + win])
        y.append(arr[i + win])
    return np.array(X)[..., None], np.array(y)[..., None]


def train_test_split_lstm(return_input, split_ratio, win=1,
                          scaling=True, scaler="Standard"):
    """Window-based LSTM data preparation (win=1 in the notebook)."""
    r2 = np.array(return_input, dtype=float) ** 2
    X, y = make_windows(r2, win)
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if scaling:
        sc_x = StandardScaler() if scaler == "Standard" else MinMaxScaler()
        sc_y = StandardScaler() if scaler == "Standard" else MinMaxScaler()

        n_tr, w, f = X_train.shape
        X_train = sc_x.fit_transform(X_train.reshape(n_tr * w, f)).reshape(n_tr, w, f)
        n_te = X_test.shape[0]
        X_test = sc_x.transform(X_test.reshape(n_te * w, f)).reshape(n_te, w, f)
        y_train = sc_y.fit_transform(y_train)
        y_test = sc_y.transform(y_test)

        return X_train, X_test, y_train, y_test, sc_x, sc_y

    return X_train, X_test, y_train, y_test


def sequential_preprocess(train_returns, test_returns, reshape_dim=2):
    """
    Prepare lagged (X, y) pairs for the sequential residual models.
    X[t] = r_{t}, y[t] = r_{t+1}  (lagged by 1).
    """
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    train_arr = np.asarray(train_returns).ravel()
    test_arr = np.asarray(test_returns).ravel()

    if reshape_dim == 2:
        X_tr = train_arr[:-1].reshape(-1, 1)
        y_tr = train_arr[1:].reshape(-1, 1)
        X_te = test_arr[:-1].reshape(-1, 1)
        y_te = test_arr[1:].reshape(-1, 1)

        X_tr_s = sc_x.fit_transform(X_tr)
        y_tr_s = sc_y.fit_transform(y_tr)
        X_te_s = sc_x.transform(X_te)
        y_te_s = sc_y.transform(y_te)

    elif reshape_dim == 3:
        X_tr = train_arr[:-1].reshape(-1, 1, 1)
        y_tr = train_arr[1:].reshape(-1, 1)
        X_te = test_arr[:-1].reshape(-1, 1, 1)
        y_te = test_arr[1:].reshape(-1, 1)

        X_tr_s = sc_x.fit_transform(X_tr.reshape(-1, 1)).reshape(-1, 1, 1)
        y_tr_s = sc_y.fit_transform(y_tr)
        X_te_s = sc_x.transform(X_te.reshape(-1, 1)).reshape(-1, 1, 1)
        y_te_s = sc_y.transform(y_te)

    return X_tr_s, y_tr_s, X_te_s, y_te_s, sc_x, sc_y


# ── Forecast Metrics ──────────────────────────────────────────────────────────

def forecast_metrics(y_test, preds, model, round_decimal=4, qlike_only=False):
    """Compute RMSE, MAE, MAPE, QLIKE. Returns scalar (qlike_only) or list."""
    eps = 1e-8
    preds = np.asarray(preds, dtype=float).copy()
    y_test = np.asarray(y_test, dtype=float).copy()

    preds = np.maximum(preds, eps)
    y_test = np.maximum(y_test, eps)

    if qlike_only:
        return float(np.round(np.mean(np.log(preds) + y_test / preds), round_decimal))

    rmse = float(np.round(np.sqrt(mean_squared_error(y_test, preds)), round_decimal))
    mae = float(np.round(mean_absolute_error(y_test, preds), round_decimal))
    mape = float(np.round(np.mean(np.abs((y_test - preds) / y_test)) * 100, round_decimal))
    qlike = float(np.round(np.mean(np.log(preds) + y_test / preds), round_decimal))
    return [model, rmse, mae, mape, qlike]


# ── Multi-step Prediction Helpers ─────────────────────────────────────────────

def nn_multi_step_preds(model, test_returns, X_test, scaler_y, H,
                        model_name, scaler_x=None, lower_triangular=False,
                        progress_cb=None):
    """
    Iterative H-step ahead forecasts for a Keras MLP.
    Returns (preds_df, qlike_array).
    """
    N = len(test_returns)
    preds_lt = np.full((N, H), np.nan, dtype=np.float64)
    x_scaled = np.asarray(X_test, dtype=np.float32).reshape(-1, 1)
    if scaler_x is None:
        scaler_x = scaler_y

    for h in range(1, H + 1):
        count = (N - h) if lower_triangular else N
        if count <= 0:
            break

        x_batch = x_scaled[:count].reshape(count, 1).astype(np.float32)
        yhat_s = model.predict(x_batch, verbose=0).reshape(-1, 1)
        yhat_s = np.nan_to_num(yhat_s.astype(np.float64), nan=0.0, posinf=6, neginf=-6)
        yhat_s = np.clip(yhat_s, -6, 6)

        yhat = scaler_y.inverse_transform(yhat_s).ravel()
        preds_lt[:count, h - 1] = yhat

        x_scaled = scaler_x.transform(scaler_y.inverse_transform(yhat_s)).astype(np.float32).reshape(-1, 1)

        if progress_cb:
            progress_cb(h, H)

    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    preds_df = pd.DataFrame(preds_lt, index=test_returns.index, columns=cols)

    qlike_arr = []
    for i in range(H):
        col_preds = preds_df.iloc[:, i].dropna().values
        proxy = (test_returns ** 2).values[:len(col_preds)]
        if len(col_preds) == 0:
            break
        qlike_arr.append(forecast_metrics(proxy, col_preds, model_name, qlike_only=True))

    return preds_df, np.array(qlike_arr)


def lstm_multi_step_preds(model, test_returns, X_test_seq, scaler_x, scaler_y,
                          H, model_name, lower_triangular=False, progress_cb=None):
    """
    Iterative H-step ahead forecasts for a Keras LSTM (win=1).
    Returns (preds_df, qlike_array).
    """
    N, seq_len, feat = X_test_seq.shape
    preds_lt = np.full((N, H), np.nan, dtype=np.float64)
    seqs = X_test_seq.astype(np.float32).copy()

    for h in range(1, H + 1):
        count = (N - h) if lower_triangular else N
        if count <= 0:
            break

        yhat_s = model.predict(seqs[:count], verbose=0).reshape(-1, 1)
        yhat = scaler_y.inverse_transform(yhat_s).ravel()
        preds_lt[:count, h - 1] = yhat

        yhat_for_x = scaler_x.transform(yhat.reshape(-1, 1)).reshape(-1, 1, 1)
        seqs = np.concatenate([seqs[:, 1:, :],
                               np.zeros((N, 1, 1), dtype=np.float32)], axis=1)
        seqs[:count, -1:, :] = yhat_for_x

        if progress_cb:
            progress_cb(h, H)

    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    preds_df = pd.DataFrame(preds_lt, index=test_returns.index, columns=cols)

    qlike_arr = []
    for i in range(H):
        col_preds = preds_df.iloc[:, i].dropna().values
        proxy = (test_returns ** 2).values[:len(col_preds)]
        if len(col_preds) == 0:
            break
        qlike_arr.append(forecast_metrics(proxy, col_preds, model_name, qlike_only=True))

    return preds_df, np.array(qlike_arr)


def multi_step_preds_torch(model, model_name, test_returns, H=100,
                           lower_triangular=False, progress_cb=None):
    """
    Iterative H-step ahead forecasts for a PyTorch AugmentedHybridModel.
    Returns (preds_df, qlike_array).
    """
    import torch
    N = len(test_returns)
    r_test = torch.tensor(test_returns.values, dtype=torch.float32).view(-1, 1)
    r = r_test.clone()
    r2 = r.pow(2)
    preds = np.full((N, H), np.nan, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for h in range(1, H + 1):
            count = (N - h) if lower_triangular else N
            if count <= 0:
                break

            r_in = r[:count].view(-1)
            r2_in = r2[:count].view(-1)
            v_hat = model(r_in, r2_in).view(-1, 1)
            v_np = v_hat.squeeze(-1).cpu().numpy()
            preds[:count, h - 1] = np.maximum(v_np, 0.0)

            r = torch.zeros_like(v_hat)
            r2 = v_hat

            if progress_cb:
                progress_cb(h, H)

    cols = [f"h.{k:03d}" for k in range(1, H + 1)]
    preds_df = pd.DataFrame(preds, index=test_returns.index, columns=cols)

    qlike_arr = []
    for i in range(H):
        col_preds = preds_df.iloc[:, i].dropna().values
        proxy = (test_returns ** 2).values[:len(col_preds)]
        if len(col_preds) == 0:
            break
        qlike_arr.append(forecast_metrics(proxy, col_preds, model_name, qlike_only=True))

    return preds_df, np.array(qlike_arr)


def comp_nstep_qlike_rmse(preds_matrix, test_proxy, H):
    """
    Compute QLIKE and RMSE arrays for h = 1..H.
    preds_matrix: DataFrame with columns h.001 .. h.H, rows = test origins.
    test_proxy: Series of squared returns (same length as preds_matrix rows).
    Returns (qlike_array, rmse_array) each of length H.
    """
    proxy = np.asarray(test_proxy.values if hasattr(test_proxy, 'values') else test_proxy,
                       dtype=float)
    qlikes, rmses = [], []
    for i in range(H):
        col = preds_matrix.iloc[:, i].values.astype(float)
        # align lengths
        n = min(len(col), len(proxy))
        p, y = col[:n], proxy[:n]
        mask = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
        if mask.sum() == 0:
            qlikes.append(np.nan)
            rmses.append(np.nan)
            continue
        p, y = p[mask], y[mask]
        qlikes.append(float(np.mean(np.log(p) + y / p)))
        rmses.append(float(np.sqrt(np.mean((p - y) ** 2))))
    return np.array(qlikes), np.array(rmses)
