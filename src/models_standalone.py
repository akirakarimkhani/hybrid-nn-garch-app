"""
Standalone benchmark and neural-network models.
Implements: GARCH(1,1), EGARCH(1,1) [optional], MLP, LSTM,
            Sequential-MLP, Sequential-LSTM.
"""
import numpy as np
import pandas as pd

# ── EGARCH availability ───────────────────────────────────────────────────────
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    EGARCH_AVAILABLE = True
except Exception:
    EGARCH_AVAILABLE = False


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def fit_garch(returns: pd.Series, H: int = 100):
    """
    Fit GARCH(1,1) on the full returns series.
    Returns (model, preds_target_df, preds_origin_df, onestep_series, params_dict).
    """
    import os
    os.environ.setdefault("ARCH_DISABLE_NUMBA", "1")
    from arch import arch_model

    split = int(len(returns) * 0.8)
    train_returns = returns.iloc[:split]
    test_index = returns.index[split:]

    garch = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1).fit(disp='off')

    preds_target = garch.forecast(
        horizon=H, start=train_returns.index[-1],
        reindex=True, align="target"
    ).variance

    preds_origin = garch.forecast(
        horizon=H, start=train_returns.index[-1],
        reindex=True, align="origin"
    ).variance

    # Rename columns to canonical h.001 .. h.H format regardless of arch version
    def _rename_cols(df, H):
        df = df.copy()
        df.columns = [f"h.{k:03d}" for k in range(1, len(df.columns) + 1)]
        return df

    preds_target = _rename_cols(preds_target, H)
    preds_origin = _rename_cols(preds_origin, H)

    # One-step-ahead forecasts for test set (first column of target-aligned df)
    target_test = preds_target.reindex(test_index)
    onestep = target_test.iloc[:, 0].dropna()

    params = {
        "Omega": float(garch.params.get("omega", np.nan)),
        "Alpha": float(garch.params.get("alpha[1]", np.nan)),
        "Beta": float(garch.params.get("beta[1]", np.nan)),
    }
    uncond_var = params["Omega"] / max(1 - params["Alpha"] - params["Beta"], 1e-8)
    params["Uncond. Var"] = float(uncond_var)

    return garch, preds_target, preds_origin, onestep, params


# ── EGARCH(1,1) ───────────────────────────────────────────────────────────────

def fit_egarch(returns: pd.Series, H: int = 100, scaling_factor: float = 1.0):
    """
    Fit EGARCH(1,1) via rpy2 + rugarch. Returns None if rpy2 unavailable.
    Returns (preds_origin_df, onestep_series, params_dict) or None.
    """
    if not EGARCH_AVAILABLE:
        return None

    split = int(len(returns) * 0.8)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]

    train_scaled = train_returns * scaling_factor
    test_scaled = test_returns * scaling_factor

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["train_returns"] = ro.conversion.py2rpy(train_scaled)
        ro.globalenv["test_returns"] = ro.conversion.py2rpy(test_scaled)
    ro.globalenv["H"] = ro.IntVector([int(H)])
    ro.globalenv["n_test"] = ro.IntVector([int(len(test_returns))])

    r_script = """
    suppressMessages(library(rugarch))
    full <- c(train_returns, test_returns)
    outsmpl <- n_test[1]
    Hh <- H[1]
    spec <- ugarchspec(
      variance.model = list(model="eGARCH", garchOrder=c(1,1)),
      mean.model = list(armaOrder=c(0,0), include.mean=FALSE),
      distribution.model = "norm"
    )
    fit <- ugarchfit(spec, data=full, out.sample=outsmpl)
    fc <- ugarchforecast(fit, n.ahead=Hh, n.roll=outsmpl)
    var_by_origin <- t((fc@forecast$sigmaFor)^2)
    colnames(var_by_origin) <- sprintf("h.%03d", seq_len(ncol(var_by_origin)))

    # Extract parameters
    coefs <- coef(fit)
    list(forecasts=as.data.frame(var_by_origin), params=coefs)
    """

    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            result = ro.r(r_script)
        df = result.rx2("forecasts")
        coefs = dict(zip(result.rx2("params").names, list(result.rx2("params"))))

        s2 = float(scaling_factor) ** 2
        df = df.astype(float) / s2
        df = df.iloc[1:, :]
        test_idx = list(test_returns.index)
        df.index = pd.Index(test_idx, name="origin")

        onestep = df["h.001"]

        params = {
            "Omega": float(coefs.get("mu", coefs.get("omega", np.nan))),
            "Alpha": float(coefs.get("alpha1", np.nan)),
            "Beta": float(coefs.get("beta1", np.nan)),
            "Gamma": float(coefs.get("gamma1", np.nan)),
        }
        return df, onestep, params
    except Exception as e:
        print(f"EGARCH fitting failed: {e}")
        return None


# ── MLP ───────────────────────────────────────────────────────────────────────

def fit_mlp(returns: pd.Series, split_ratio: float = 0.8,
            epochs: int = 100, learning_rate: float = 0.001,
            batch_size: int = 32, validation_split: float = 0.1):
    """
    Train a 3-layer MLP on squared returns.
    Returns (model, scaler_x, scaler_y, loss_history).
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    from src.utils import gaussian_nll, train_test_split_nn

    X_train, X_test, y_train, y_test, scaler_x, scaler_y = train_test_split_nn(
        returns, split_ratio, scaling=True, scaler="Standard", reshape_dim=2
    )

    model = Sequential([
        Dense(16, activation='relu', input_shape=(1,)),
        Dense(8, activation='relu'),
        Dense(1, activation='softplus'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=gaussian_nll)
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=0)

    return model, scaler_x, scaler_y, hist.history['loss']


# ── LSTM ──────────────────────────────────────────────────────────────────────

def fit_lstm(returns: pd.Series, split_ratio: float = 0.8,
             epochs: int = 100, learning_rate: float = 0.001,
             batch_size: int = 32, validation_split: float = 0.1):
    """
    Train a LSTM on squared returns (win=1).
    Returns (model, scaler_x, scaler_y, loss_history).
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from src.utils import train_test_split_lstm

    X_train, X_test, y_train, y_test, scaler_x, scaler_y = train_test_split_lstm(
        returns, split_ratio, win=1, scaling=True, scaler="Standard"
    )

    model = Sequential([
        LSTM(16, input_shape=(1, 1)),
        Dense(8, activation='relu'),
        Dense(1, activation='softplus'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=0)

    return model, scaler_x, scaler_y, hist.history['loss']


# ── Sequential Residual MLP ───────────────────────────────────────────────────

def fit_sequential_mlp(train_returns: pd.Series, test_returns: pd.Series,
                       split_ratio: float = 0.8, H: int = 100,
                       epochs: int = 100, learning_rate: float = 0.001,
                       batch_size: int = 32, validation_split: float = 0.1):
    """
    GARCH(1,1) → MLP: train MLP on GARCH conditional variance forecasts.
    Returns (model, scaler_x, scaler_y, X_test_scaled, loss_history, onestep_garch_preds).
    """
    from arch import arch_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    from src.utils import gaussian_nll, sequential_preprocess

    garch_res = arch_model(train_returns, mean='Zero', vol='GARCH', p=1, q=1).fit(disp='off')
    garch_full_preds = garch_res.forecast(
        horizon=H, start=train_returns.index[0],
        reindex=True, align="origin"
    ).variance.iloc[:, 0]  # first column = 1-step ahead variance

    garch_test_res = arch_model(test_returns, mean='Zero', vol='GARCH', p=1, q=1).fit(disp='off')
    garch_test_preds = garch_test_res.forecast(
        horizon=H, start=train_returns.index[0],
        reindex=True, align="origin"
    ).variance.iloc[:, 0]

    train_sq = train_returns ** 2
    X_tr, y_tr, X_te, y_te, sc_x, sc_y = sequential_preprocess(
        train_returns=garch_full_preds,
        test_returns=train_sq.values,
        reshape_dim=2
    )

    X_test_arr = np.array(garch_test_preds).reshape(-1, 1)
    X_test_scaled = sc_x.transform(X_test_arr)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(1,)),
        Dense(8, activation='relu'),
        Dense(1, activation='softplus'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=gaussian_nll)
    hist = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=0)

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    onestep = sc_y.inverse_transform(y_pred_scaled).flatten()
    onestep_series = pd.Series(onestep, index=test_returns.index[:len(onestep)])

    return model, sc_x, sc_y, X_test_scaled, hist.history['loss'], onestep_series


# ── Sequential Residual LSTM ──────────────────────────────────────────────────

def fit_sequential_lstm(train_returns: pd.Series, test_returns: pd.Series,
                        split_ratio: float = 0.8, H: int = 100,
                        epochs: int = 100, learning_rate: float = 0.001,
                        batch_size: int = 32, validation_split: float = 0.1):
    """
    GARCH(1,1) → LSTM: train LSTM on GARCH conditional variance forecasts.
    Returns (model, scaler_x, scaler_y, X_test_scaled, loss_history, onestep_series).
    """
    from arch import arch_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from src.utils import gaussian_nll, sequential_preprocess

    garch_res = arch_model(train_returns, mean='Zero', vol='GARCH', p=1, q=1).fit(disp='off')
    garch_full_preds = garch_res.forecast(
        horizon=H, start=train_returns.index[0],
        reindex=True, align="origin"
    ).variance.iloc[:, 0]

    garch_test_res = arch_model(test_returns, mean='Zero', vol='GARCH', p=1, q=1).fit(disp='off')
    garch_test_preds = garch_test_res.forecast(
        horizon=H, start=train_returns.index[0],
        reindex=True, align="origin"
    ).variance.iloc[:, 0]

    X_tr, y_tr, X_te, y_te, sc_x, sc_y = sequential_preprocess(
        train_returns=np.array(garch_full_preds),
        test_returns=test_returns.values,
        reshape_dim=3
    )

    X_test_arr = np.array(garch_test_preds.dropna()).reshape(-1, 1)
    X_test_scaled = sc_x.transform(X_test_arr).reshape(-1, 1, 1)

    model = Sequential([
        LSTM(16, input_shape=(1, 1)),
        Dense(8, activation='relu'),
        Dense(1, activation='softplus'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=gaussian_nll)
    hist = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=0)

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    onestep = sc_y.inverse_transform(y_pred_scaled).flatten()
    onestep_series = pd.Series(onestep, index=test_returns.index[:len(onestep)])

    return model, sc_x, sc_y, X_test_scaled, hist.history['loss'], onestep_series
