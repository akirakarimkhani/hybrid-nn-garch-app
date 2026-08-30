"""
Matplotlib figure builders for the Streamlit app.
All functions return a matplotlib.figure.Figure.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats as scipy_stats


# -- Colours / styles ----------------------------------------------------------
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
MODEL_COLORS = {
    "GARCH": "#1f77b4",
    "EGARCH": "#ff7f0e",
    "MLP": "#2ca02c",
    "LSTM": "#d62728",
    "Aug-Donaldson": "#9467bd",
    "Aug-MLP": "#8c564b",
    "Aug-LSTM": "#e377c2",
    "Seq-MLP": "#7f7f7f",
    "Seq-LSTM": "#bcbd22",
    "Int-MLP": "#17becf",
    "Int-LSTM": "#aec7e8",
}


def _close(fig):
    plt.close(fig)
    return fig


# -- Data Overview -------------------------------------------------------------

def plot_prices_and_returns(prices: pd.Series, returns: pd.Series,
                            ticker: str = "") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    ax1.plot(prices.index.to_numpy(), prices.values, linewidth=0.8, color="#1f77b4")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} - Closing Price")
    ax1.tick_params(axis='x', rotation=30)

    ax2.plot(returns.index.to_numpy(), returns.values, linewidth=0.5, color="#d62728", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Log Return x 100")
    ax2.set_title("Daily Log Returns (x100)")
    ax2.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    return _close(fig)


def plot_qq(returns: pd.Series, ticker: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    scipy_stats.probplot(returns.dropna(), dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot - {ticker} Log Returns")
    fig.tight_layout()
    return _close(fig)


def plot_acf_pacf(returns: pd.Series, lags: int = 20) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plot_acf(returns, lags=lags, ax=axes[0, 0], alpha=0.05)
    axes[0, 0].set_title("ACF - Returns")
    plot_pacf(returns, lags=lags, ax=axes[0, 1], method='ywm', alpha=0.05)
    axes[0, 1].set_title("PACF - Returns")
    plot_acf(returns ** 2, lags=lags, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title("ACF - Squared Returns")
    plot_pacf(returns ** 2, lags=lags, ax=axes[1, 1], method='ywm', alpha=0.05)
    axes[1, 1].set_title("PACF - Squared Returns")
    fig.tight_layout()
    return _close(fig)


# -- Forecast Plots ------------------------------------------------------------

def plot_onestep_forecasts(onestep_preds: dict,
                           test_sq_returns: pd.Series,
                           title_suffix: str = "") -> plt.Figure:
    """
    Multi-panel plot of one-step-ahead σ² forecasts for each model family.
    onestep_preds: {model_key: pd.Series}
    """
    n = len(onestep_preds)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No models trained yet.", ha='center', va='center')
        return _close(fig)

    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    ax_flat = axes.flatten()

    proxy_vals = test_sq_returns.values
    proxy_idx = test_sq_returns.index.to_numpy()

    for idx, (key, preds) in enumerate(onestep_preds.items()):
        ax = ax_flat[idx]
        ax.plot(proxy_idx, proxy_vals, color="lightgrey", label="r²_t", linewidth=0.7, zorder=1)
        if preds is not None:
            n_p = min(len(preds), len(proxy_idx))
            ax.plot(proxy_idx[:n_p], preds.values[:n_p],
                    color=MODEL_COLORS.get(key, "#333333"),
                    label=key, linewidth=0.9, linestyle="--", zorder=2)
        ax.set_title(f"{key} - One-step σ²")
        ax.set_xlabel("Date")
        ax.set_ylabel("Conditional Variance")
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=30)

    for i in range(len(onestep_preds), len(ax_flat)):
        ax_flat[i].set_visible(False)

    fig.suptitle(f"One-Step Ahead Forecasts vs Squared Returns{title_suffix}", fontsize=12)
    fig.tight_layout()
    return _close(fig)


def plot_multistep_qlike(qlike_arrays: dict, H: int) -> plt.Figure:
    """
    QLIKE vs forecast horizon h=1..H for each model.
    qlike_arrays: {model_key: np.array of length H}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, arr in qlike_arrays.items():
        if arr is None or len(arr) == 0:
            continue
        h_range = np.arange(1, len(arr) + 1)
        ax.plot(h_range, arr, label=key, color=MODEL_COLORS.get(key),
                linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Forecast Horizon h")
    ax.set_ylabel("QLIKE Loss")
    ax.set_title("Multi-Step QLIKE vs. Horizon (Squared Returns Proxy)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _close(fig)


def plot_multistep_rmse(rmse_arrays: dict, H: int) -> plt.Figure:
    """RMSE vs forecast horizon for each model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, arr in rmse_arrays.items():
        if arr is None or len(arr) == 0:
            continue
        h_range = np.arange(1, len(arr) + 1)
        ax.plot(h_range, arr, label=key, color=MODEL_COLORS.get(key),
                linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Forecast Horizon h")
    ax.set_ylabel("RMSE")
    ax.set_title("Multi-Step RMSE vs. Horizon (Squared Returns Proxy)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _close(fig)


def plot_correlation_heatmap(onestep_preds: dict,
                             test_sq_returns: pd.Series) -> plt.Figure:
    """Seaborn heatmap of pairwise Spearman correlations between model forecasts."""
    frames = {}
    proxy_len = len(test_sq_returns)
    for k, v in onestep_preds.items():
        if v is not None:
            frames[k] = v.values[:proxy_len] if hasattr(v, 'values') else v[:proxy_len]

    if len(frames) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need ≥ 2 models for correlation matrix.", ha='center', va='center')
        return _close(fig)

    df = pd.DataFrame(frames)
    corr = df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(max(6, len(frames)), max(5, len(frames) - 1)))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                mask=mask, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Spearman Correlation of One-Step-Ahead σ² Forecasts")
    fig.tight_layout()
    return _close(fig)


def plot_dm_heatmap(dm_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of DM test p-values."""
    fig, ax = plt.subplots(figsize=(max(6, len(dm_df)), max(5, len(dm_df) - 1)))
    sns.heatmap(dm_df.astype(float), annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=0.1, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("DM Test p-values (QLIKE, two-sided)\nGreen = significant difference at 10%")
    fig.tight_layout()
    return _close(fig)


def plot_training_losses(loss_histories: dict) -> plt.Figure:
    """Training loss curves for neural-network models."""
    valid = {k: v for k, v in loss_histories.items() if v is not None and len(v) > 0}
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No training curves available.", ha='center', va='center')
        return _close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, loss in valid.items():
        ax.plot(np.arange(1, len(loss) + 1), loss,
                label=key, color=MODEL_COLORS.get(key), linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _close(fig)


def plot_params_table(params_dict: dict) -> plt.Figure:
    """Render the parameters table as a matplotlib figure."""
    if not params_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameters available.", ha='center', va='center')
        return _close(fig)

    rows = []
    for model, params in params_dict.items():
        for param_name, val in params.items():
            rows.append({"Model": model, "Parameter": param_name, "Value": round(val, 6)})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(df) + 1)))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(True)
    tbl.scale(1, 1.4)
    ax.set_title("Fitted Model Parameters", pad=12)
    fig.tight_layout()
    return _close(fig)
