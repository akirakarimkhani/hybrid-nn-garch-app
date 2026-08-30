"""
Volatility Forecasting Comparison Tool
Streamlit app implementing 11 GARCH / hybrid GARCH-NN models from the
Master's thesis: "A Systematic Formalization of Hybrid GARCH-Neural Network
Approaches for Volatility Forecasting" (HU Berlin).
"""
import os, random
os.environ["PYTHONHASHSEED"] = "10"
os.environ["ARCH_DISABLE_NUMBA"] = "1"   # arch's numba JIT is broken in this env
random.seed(10)

import numpy as np
np.random.seed(10)

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date, timedelta

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Volatility Forecasting Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Local imports (deferred to avoid TF/Torch import noise at startup) --------
from src.data import load_stock_data, train_test_split_returns, descriptive_stats
from src.models_standalone import (
    EGARCH_AVAILABLE, fit_garch, fit_egarch, fit_mlp, fit_lstm,
    fit_sequential_mlp, fit_sequential_lstm,
)
from src.utils import (
    train_test_split_nn, train_test_split_lstm,
    nn_multi_step_preds, lstm_multi_step_preds, multi_step_preds_torch,
    comp_nstep_qlike_rmse,
)
from src.evaluation import build_metrics_table, dm_pairwise_table
from src.visualization import (
    plot_prices_and_returns, plot_qq, plot_acf_pacf,
    plot_onestep_forecasts, plot_multistep_qlike, plot_multistep_rmse,
    plot_correlation_heatmap, plot_dm_heatmap,
    plot_training_losses, plot_params_table,
)

# -- Sidebar -------------------------------------------------------------------
st.sidebar.title("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", value=date(2015, 1, 1))
end_date = col2.date_input("End", value=date.today() - timedelta(days=1))
train_ratio = st.sidebar.slider("Train / Test split", 0.6, 0.9, 0.8, 0.05)
H = st.sidebar.slider("Forecast horizon H", 10, 100, 100, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Families")
run_econ = st.sidebar.checkbox("Econometric (GARCH, EGARCH)", value=True)
run_nn = st.sidebar.checkbox("Standalone NN (MLP, LSTM)", value=True)
run_aug = st.sidebar.checkbox("Augmented Hybrid", value=True)
run_seq = st.sidebar.checkbox("Sequential Hybrid", value=True)
run_int = st.sidebar.checkbox("Integrated Hybrid", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")
epochs = st.sidebar.number_input("Epochs (NN models)", min_value=10, max_value=500,
                                  value=100, step=10)
lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=0.1,
                              value=0.001, format="%.5f")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)

if not EGARCH_AVAILABLE:
    st.sidebar.info("EGARCH requires R + rugarch. Model skipped.")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Analysis", use_container_width=True, type="primary")

# -- Main content --------------------------------------------------------------
st.title("Volatility Forecasting Model Comparison")
st.markdown(
    "Compare **11 volatility models** across econometric benchmarks, "
    "standalone neural networks, and three families of hybrid GARCH-NN architectures."
)

# -- Data loading --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load(ticker, start, end):
    return load_stock_data(ticker, str(start), str(end))


def get_data():
    with st.spinner(f"Downloading {ticker} data..."):
        prices, returns = cached_load(ticker, start_date, end_date)
    return prices, returns


# -- Training orchestrator -----------------------------------------------------
def run_analysis(returns, train_returns, test_returns, status_container):
    """
    Fit all selected models and return:
      results = {
        "onestep":   {model_key: pd.Series},
        "multistep": {model_key: pd.DataFrame (N_test x H)},
        "qlike":     {model_key: np.array},
        "rmse":      {model_key: np.array},
        "losses":    {model_key: list},
        "params":    {model_key: dict},
      }
    """
    import torch
    import tensorflow as tf
    tf.random.set_seed(10)
    torch.manual_seed(10)

    results = {
        "onestep": {}, "multistep": {}, "qlike": {}, "rmse": {}, "losses": {}, "params": {}
    }
    test_sq = test_returns ** 2

    def _store(key, onestep, ms_df, losses=None, params=None):
        results["onestep"][key] = onestep
        results["multistep"][key] = ms_df
        if ms_df is not None:
            ql, rm = comp_nstep_qlike_rmse(ms_df, test_sq, H)
            results["qlike"][key] = ql
            results["rmse"][key] = rm
        if losses:
            results["losses"][key] = losses
        if params:
            results["params"][key] = params

    # -- GARCH -----------------------------------------------------------------
    if run_econ:
        with status_container.status("Fitting GARCH(1,1)...", expanded=False) as s:
            try:
                _, preds_target, preds_origin, onestep, params = fit_garch(returns, H=H)
                # Build N_test x H matrix from origin-aligned forecasts
                test_idx = test_returns.index
                ms_df = preds_origin.reindex(test_idx)
                # Ensure exactly H columns
                existing_cols = ms_df.shape[1]
                if existing_cols < H:
                    for k in range(existing_cols + 1, H + 1):
                        ms_df[f"h.{k:03d}"] = np.nan
                ms_df = ms_df.iloc[:, :H]
                ms_df.columns = [f"h.{k:03d}" for k in range(1, H + 1)]
                onestep = onestep.reindex(test_idx)
                _store("GARCH", onestep, ms_df, params=params)
                s.update(label="GARCH(1,1) done", state="complete")
            except Exception as e:
                import traceback
                s.update(label=f"GARCH failed: {e}", state="error")
                st.error(traceback.format_exc())

    # -- EGARCH ----------------------------------------------------------------
    if run_econ and EGARCH_AVAILABLE:
        with status_container.status("Fitting EGARCH(1,1)...", expanded=False) as s:
            try:
                res = fit_egarch(returns, H=H, scaling_factor=1.0)
                if res is not None:
                    df_eg, onestep_eg, params_eg = res
                    _store("EGARCH", onestep_eg, df_eg, params=params_eg)
                    s.update(label="EGARCH(1,1) done", state="complete")
                else:
                    s.update(label="EGARCH returned no results", state="error")
            except Exception as e:
                s.update(label=f"EGARCH failed: {e}", state="error")

    # -- MLP -------------------------------------------------------------------
    if run_nn:
        with status_container.status("Training MLP...", expanded=False) as s:
            try:
                model_mlp, sc_x, sc_y, loss_mlp = fit_mlp(
                    returns, train_ratio, epochs, lr, batch_size
                )
                X_train, X_test, y_train, y_test, _, _ = train_test_split_nn(
                    returns, train_ratio, scaling=True
                )
                preds_s = model_mlp.predict(X_test, verbose=0).flatten()
                onestep_mlp = pd.Series(
                    sc_y.inverse_transform(preds_s.reshape(-1, 1)).flatten(),
                    index=test_returns.index[:len(preds_s)]
                )
                ms_df_mlp, _ = nn_multi_step_preds(
                    model_mlp, test_returns, X_test, sc_y, H, "MLP", scaler_x=sc_x
                )
                _store("MLP", onestep_mlp, ms_df_mlp, losses=loss_mlp)
                s.update(label="MLP done", state="complete")
            except Exception as e:
                s.update(label=f"MLP failed: {e}", state="error")

    # -- LSTM ------------------------------------------------------------------
    if run_nn:
        with status_container.status("Training LSTM...", expanded=False) as s:
            try:
                model_lstm, sc_x_lstm, sc_y_lstm, loss_lstm = fit_lstm(
                    returns, train_ratio, epochs, lr, batch_size
                )
                _, X_test_lstm, _, _, scx_l, scy_l = train_test_split_lstm(
                    returns, train_ratio, win=1, scaling=True
                )
                preds_s_lstm = model_lstm.predict(X_test_lstm, verbose=0).flatten()
                onestep_lstm = pd.Series(
                    scy_l.inverse_transform(preds_s_lstm.reshape(-1, 1)).flatten(),
                    index=test_returns.index[:len(preds_s_lstm)]
                )
                ms_df_lstm, _ = lstm_multi_step_preds(
                    model_lstm, test_returns, X_test_lstm,
                    scx_l, scy_l, H, "LSTM"
                )
                _store("LSTM", onestep_lstm, ms_df_lstm, losses=loss_lstm)
                s.update(label="LSTM done", state="complete")
            except Exception as e:
                s.update(label=f"LSTM failed: {e}", state="error")

    # -- Augmented: Donaldson --------------------------------------------------
    if run_aug:
        with status_container.status("Training Augmented-Donaldson...", expanded=False) as s:
            try:
                from src.models_augmented import train_aug_donaldson, multi_step_preds_torch as aug_ms
                model_don, loss_don, params_don = train_aug_donaldson(train_returns, epochs=epochs)
                ms_df_don, _ = multi_step_preds_torch(model_don, "Aug-Donaldson", test_returns, H=H)
                onestep_don = ms_df_don["h.001"]
                _store("Aug-Donaldson", onestep_don, ms_df_don, losses=loss_don, params=params_don)
                s.update(label="Augmented-Donaldson done", state="complete")
            except Exception as e:
                s.update(label=f"Aug-Donaldson failed: {e}", state="error")

    # -- Augmented: Extended MLP -----------------------------------------------
    if run_aug:
        with status_container.status("Training Augmented-MLP...", expanded=False) as s:
            try:
                from src.models_augmented import (
                    AugmentedModel_Extended_mlp, run_joint_training, aug_mlp_multistep_preds
                )
                net_aug, loss_aug, params_aug, gp_aug = run_joint_training(
                    AugmentedModel_Extended_mlp(), train_returns, epochs=epochs
                )
                ms_df_aug, _ = aug_mlp_multistep_preds(
                    net_aug, gp_aug, test_returns, H=H
                )
                onestep_aug = ms_df_aug["h.001"]
                _store("Aug-MLP", onestep_aug, ms_df_aug, losses=loss_aug, params=params_aug)
                s.update(label="Augmented-MLP done", state="complete")
            except Exception as e:
                s.update(label=f"Aug-MLP failed: {e}", state="error")

    # -- Augmented: Extended LSTM ----------------------------------------------
    if run_aug:
        with status_container.status("Training Augmented-LSTM...", expanded=False) as s:
            try:
                from src.models_augmented import run_lstm_joint_training, aug_lstm_multistep_preds
                net_alstm, gp_alstm, loss_alstm, params_alstm = run_lstm_joint_training(
                    train_returns, epochs=epochs
                )
                ms_df_alstm, _ = aug_lstm_multistep_preds(
                    net_alstm, gp_alstm, test_returns, H=H
                )
                onestep_alstm = ms_df_alstm["h.001"]
                _store("Aug-LSTM", onestep_alstm, ms_df_alstm,
                       losses=loss_alstm, params=params_alstm)
                s.update(label="Augmented-LSTM done", state="complete")
            except Exception as e:
                s.update(label=f"Aug-LSTM failed: {e}", state="error")

    # -- Sequential: MLP -------------------------------------------------------
    if run_seq:
        with status_container.status("Training Sequential-MLP...", expanded=False) as s:
            try:
                model_sml, sc_x_sml, sc_y_sml, X_test_sml, loss_sml, onestep_sml = \
                    fit_sequential_mlp(train_returns, test_returns,
                                       train_ratio, H, epochs, lr, batch_size)
                ms_df_sml, _ = nn_multi_step_preds(
                    model_sml, test_returns,
                    X_test_sml, sc_y_sml, H, "Seq-MLP"
                )
                _store("Seq-MLP", onestep_sml, ms_df_sml, losses=loss_sml)
                s.update(label="Sequential-MLP done", state="complete")
            except Exception as e:
                s.update(label=f"Seq-MLP failed: {e}", state="error")

    # -- Sequential: LSTM ------------------------------------------------------
    if run_seq:
        with status_container.status("Training Sequential-LSTM...", expanded=False) as s:
            try:
                model_sls, sc_x_sls, sc_y_sls, X_test_sls, loss_sls, onestep_sls = \
                    fit_sequential_lstm(train_returns, test_returns,
                                        train_ratio, H, epochs, lr, batch_size)
                # X_test_sls is (N, 1, 1) - use lstm_multi_step_preds
                ms_df_sls, _ = lstm_multi_step_preds(
                    model_sls, test_returns, X_test_sls,
                    sc_x_sls, sc_y_sls, H, "Seq-LSTM"
                )
                _store("Seq-LSTM", onestep_sls, ms_df_sls, losses=loss_sls)
                s.update(label="Sequential-LSTM done", state="complete")
            except Exception as e:
                s.update(label=f"Seq-LSTM failed: {e}", state="error")

    # -- Integrated: MLP -------------------------------------------------------
    if run_int:
        with status_container.status("Training Integrated-MLP...", expanded=False) as s:
            try:
                from src.models_integrated import train_integrated_mlp, integrated_mlp_multistep_preds
                model_iml, sigma_test_iml, loss_iml, params_iml, ret_tensor = \
                    train_integrated_mlp(returns, train_ratio, epochs, lr, warmup_window=20)
                onestep_iml = pd.Series(
                    sigma_test_iml.detach().numpy(),
                    index=test_returns.index[:len(sigma_test_iml)]
                )
                ms_df_iml, _ = integrated_mlp_multistep_preds(
                    model_iml, ret_tensor, train_ratio, H,
                    warmup_window=20, test_index=test_returns.index
                )
                _store("Int-MLP", onestep_iml, ms_df_iml, losses=loss_iml, params=params_iml)
                s.update(label="Integrated-MLP done", state="complete")
            except Exception as e:
                s.update(label=f"Int-MLP failed: {e}", state="error")

    # -- Integrated: LSTM ------------------------------------------------------
    if run_int:
        with status_container.status("Training Integrated-LSTM...", expanded=False) as s:
            try:
                from src.models_integrated import (
                    train_integrated_lstm, integrated_lstm_multistep_preds
                )
                model_ils, sigma_init_ils, c_init_ils, loss_ils, params_ils = \
                    train_integrated_lstm(train_returns, epochs=epochs, lr=0.01)
                ms_df_ils, _ = integrated_lstm_multistep_preds(
                    model_ils, test_returns,
                    sigma_init_ils, c_init_ils, H=H
                )
                onestep_ils = ms_df_ils["h.001"]
                _store("Int-LSTM", onestep_ils, ms_df_ils,
                       losses=loss_ils, params=params_ils)
                s.update(label="Integrated-LSTM done", state="complete")
            except Exception as e:
                s.update(label=f"Int-LSTM failed: {e}", state="error")

    return results


# -- Trigger data load on first render or when ticker/dates change -------------
if "prices" not in st.session_state or st.session_state.get("_last_query") != (ticker, start_date, end_date):
    try:
        prices, returns = get_data()
        st.session_state["prices"] = prices
        st.session_state["returns"] = returns
        st.session_state["_last_query"] = (ticker, start_date, end_date)
        st.session_state.pop("results", None)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

prices = st.session_state["prices"]
returns = st.session_state["returns"]
train_returns, test_returns = train_test_split_returns(returns, train_ratio)

# -- Run button ----------------------------------------------------------------
if run_btn:
    st.session_state.pop("results", None)
    status = st.empty()
    with status.container():
        st.markdown(f"### Training models on **{ticker}** "
                    f"({str(start_date)} to {str(end_date)}) ...")
        st.caption(f"Train: {len(train_returns)} obs | Test: {len(test_returns)} obs "
                   f"| H={H} | Epochs={epochs}")
        status_box = st.container()
        results = run_analysis(returns, train_returns, test_returns, status_box)
        st.session_state["results"] = results
    st.success("All models trained. See tabs below.")

# -- Tabs ----------------------------------------------------------------------
tab_data, tab_onestep, tab_multistep, tab_metrics, tab_stats, tab_curves, tab_params = st.tabs([
    "Data Overview",
    "One-Step Forecasts",
    "Multi-Step Performance",
    "Summary Metrics",
    "Statistical Tests",
    "Training Curves",
    "Parameters",
])

# -- TAB 1: Data Overview ------------------------------------------------------
with tab_data:
    st.subheader(f"{ticker} - Data Overview")
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = plot_prices_and_returns(prices, returns, ticker)
        st.pyplot(fig, use_container_width=True)
    with c2:
        stats = descriptive_stats(returns)
        st.markdown("**Descriptive Statistics**")
        for k, v in stats.items():
            st.metric(k, f"{v:.4f}" if isinstance(v, float) else v)

    col_qq, col_acf = st.columns(2)
    with col_qq:
        st.pyplot(plot_qq(returns, ticker), use_container_width=True)
    with col_acf:
        st.pyplot(plot_acf_pacf(returns, lags=20), use_container_width=True)

    train_r, test_r = train_test_split_returns(returns, train_ratio)
    st.caption(
        f"Train set: **{len(train_r)}** obs ({train_r.index[0].date()} - {train_r.index[-1].date()})  |  "
        f"Test set: **{len(test_r)}** obs ({test_r.index[0].date()} - {test_r.index[-1].date()})"
    )

# -- TAB 2: One-Step Forecasts -------------------------------------------------
with tab_onestep:
    st.subheader("One-Step-Ahead Conditional Variance Forecasts")
    if "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to train models.")
    else:
        res = st.session_state["results"]
        fig = plot_onestep_forecasts(
            res["onestep"], test_returns ** 2, f" - {ticker}"
        )
        st.pyplot(fig, use_container_width=True)

# -- TAB 3: Multi-Step Performance ---------------------------------------------
with tab_multistep:
    st.subheader(f"Multi-Step QLIKE and RMSE vs. Horizon h=1..{H}")
    if "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to train models.")
    else:
        res = st.session_state["results"]
        col_q, col_r = st.columns(2)
        with col_q:
            fig_q = plot_multistep_qlike(res["qlike"], H)
            st.pyplot(fig_q, use_container_width=True)
        with col_r:
            fig_r = plot_multistep_rmse(res["rmse"], H)
            st.pyplot(fig_r, use_container_width=True)

# -- TAB 4: Summary Metrics ----------------------------------------------------
with tab_metrics:
    st.subheader("Summary Metrics at h=1 (One-Step Ahead, Squared Returns Proxy)")
    if "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to train models.")
    else:
        res = st.session_state["results"]
        metrics_df = build_metrics_table(res["onestep"], test_returns ** 2)
        st.dataframe(
            metrics_df.style
                .highlight_min(subset=["QLIKE", "RMSE"], color="#c6efce")
                .highlight_max(subset=["QLIKE", "RMSE"], color="#ffc7ce")
                .format({"RMSE": "{:.6f}", "MAE": "{:.6f}",
                         "MAPE (%)": "{:.2f}", "QLIKE": "{:.6f}"}),
            use_container_width=True,
        )
        st.markdown(
            "**Green** = best (lowest) | **Red** = worst | "
            "Sorted by QLIKE (lower is better)"
        )

# -- TAB 5: Statistical Tests --------------------------------------------------
with tab_stats:
    st.subheader("Statistical Tests")
    if "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to train models.")
    else:
        res = st.session_state["results"]
        n_models = len([v for v in res["onestep"].values() if v is not None])

        st.markdown("#### Spearman Correlation of One-Step-Ahead Forecasts")
        if n_models >= 2:
            fig_corr = plot_correlation_heatmap(res["onestep"], test_returns ** 2)
            st.pyplot(fig_corr, use_container_width=True)

        st.markdown("#### Diebold-Mariano Test (QLIKE-based p-values)")
        if n_models >= 2:
            with st.spinner("Computing DM tests..."):
                dm_df = dm_pairwise_table(res["onestep"], test_returns ** 2)
            st.dataframe(
                dm_df.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.1)
                     .format("{:.3f}"),
                use_container_width=True,
            )
            fig_dm = plot_dm_heatmap(dm_df)
            st.pyplot(fig_dm, use_container_width=True)
            st.caption("p < 0.05 indicates a significant difference at the 5% level")
        else:
            st.info("Need at least 2 models for pairwise tests.")

# -- TAB 6: Training Curves ----------------------------------------------------
with tab_curves:
    st.subheader("Neural Network Training Loss Curves")
    if "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to train models.")
    else:
        res = st.session_state["results"]
        if res["losses"]:
            fig_loss = plot_training_losses(res["losses"])
            st.pyplot(fig_loss, use_container_width=True)

            st.markdown("**Final Training Losses**")
            loss_summary = {k: round(v[-1], 6) for k, v in res["losses"].items() if v}
            st.dataframe(
                pd.DataFrame(loss_summary.items(), columns=["Model", "Final Loss"]),
                use_container_width=True,
            )
        else:
            st.info("No training curves - only econometric models were selected.")

# -- TAB 7: Parameters ---------------------------------------------------------
with tab_params:
    st.subheader("Fitted Model Parameters")
    if "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to train models.")
    else:
        res = st.session_state["results"]
        if res["params"]:
            rows = []
            for model_key, params in res["params"].items():
                for pname, pval in params.items():
                    rows.append({
                        "Model": model_key,
                        "Parameter": pname,
                        "Value": round(float(pval), 6)
                    })
            params_df = pd.DataFrame(rows)
            st.dataframe(params_df, use_container_width=True)

            params_pivot = params_df.pivot_table(
                index="Parameter", columns="Model", values="Value", aggfunc="first"
            ).round(6)
            st.markdown("**Pivot View**")
            st.dataframe(params_pivot, use_container_width=True)
        else:
            st.info("No parameters available (no GARCH or hybrid models trained).")

# -- Footer --------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Master's thesis - A Systematic Formalization of Hybrid GARCH-Neural Network "
    "Approaches for Volatility Forecasting | HU Berlin | Volatility proxy: squared returns"
)
