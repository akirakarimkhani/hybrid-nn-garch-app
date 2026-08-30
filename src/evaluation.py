"""
Evaluation metrics, model comparison tables, and Diebold-Mariano tests.
"""
import numpy as np
import pandas as pd
from src.utils import forecast_metrics


MODEL_DISPLAY_NAMES = {
    "GARCH": "GARCH(1,1)",
    "EGARCH": "EGARCH(1,1)",
    "MLP": "NN: MLP",
    "LSTM": "NN: LSTM",
    "Aug-Donaldson": "Aug. Donaldson",
    "Aug-MLP": "Aug. MLP",
    "Aug-LSTM": "Aug. LSTM",
    "Seq-MLP": "Seq. MLP",
    "Seq-LSTM": "Seq. LSTM",
    "Int-MLP": "Int. MLP",
    "Int-LSTM": "Int. LSTM",
}


def build_metrics_table(onestep_preds: dict, test_sq_returns: pd.Series) -> pd.DataFrame:
    """
    Build a summary metrics table.
    onestep_preds: {model_key: pd.Series of one-step-ahead σ² forecasts}
    Returns DataFrame with columns [Model, RMSE, MAE, MAPE, QLIKE].
    """
    rows = []
    proxy = test_sq_returns.values.astype(float)

    for key, preds in onestep_preds.items():
        if preds is None:
            continue
        p = np.asarray(preds.values if hasattr(preds, 'values') else preds, dtype=float)
        n = min(len(p), len(proxy))
        p, y = p[:n], proxy[:n]
        mask = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
        if mask.sum() == 0:
            continue
        res = forecast_metrics(y[mask], p[mask], MODEL_DISPLAY_NAMES.get(key, key))
        rows.append(res)

    df = pd.DataFrame(rows, columns=["Model", "RMSE", "MAE", "MAPE (%)", "QLIKE"])
    df = df.sort_values("QLIKE").reset_index(drop=True)
    return df


def compute_qlike_rmse_arrays(preds_matrix: pd.DataFrame,
                               test_sq_returns: pd.Series,
                               H: int):
    """
    Return (qlike_array, rmse_array) of length H for a forecast matrix.
    """
    from src.utils import comp_nstep_qlike_rmse
    return comp_nstep_qlike_rmse(preds_matrix, test_sq_returns, H)


def dm_test_pair(test_proxy: np.ndarray, preds_a: np.ndarray,
                 preds_b: np.ndarray):
    """
    Diebold-Mariano test based on QLIKE loss differences.
    Returns (stat, p_value) or (nan, nan) on failure.
    """
    try:
        from dieboldmariano import dm_test
        eps = 1e-8
        pa = np.maximum(preds_a.astype(float), eps)
        pb = np.maximum(preds_b.astype(float), eps)
        y = np.maximum(test_proxy.astype(float), eps)
        n = min(len(pa), len(pb), len(y))
        pa, pb, y = pa[:n], pb[:n], y[:n]
        mask = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y)
        if mask.sum() < 4:
            return np.nan, np.nan
        stat, pval = dm_test(y[mask], pa[mask], pb[mask])
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def dm_pairwise_table(onestep_preds: dict,
                      test_sq_returns: pd.Series) -> pd.DataFrame:
    """
    Compute pairwise DM test p-values (QLIKE-based) for all model pairs.
    Returns a symmetric DataFrame of p-values.
    """
    keys = [k for k, v in onestep_preds.items() if v is not None]
    proxy = test_sq_returns.values.astype(float)
    n = len(keys)
    mat = np.full((n, n), np.nan)

    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                mat[i, j] = np.nan
                continue
            pi = np.asarray(onestep_preds[ki].values if hasattr(onestep_preds[ki], 'values')
                            else onestep_preds[ki], dtype=float)
            pj = np.asarray(onestep_preds[kj].values if hasattr(onestep_preds[kj], 'values')
                            else onestep_preds[kj], dtype=float)
            _, pval = dm_test_pair(proxy, pi, pj)
            mat[i, j] = pval

    labels = [MODEL_DISPLAY_NAMES.get(k, k) for k in keys]
    return pd.DataFrame(mat, index=labels, columns=labels)
