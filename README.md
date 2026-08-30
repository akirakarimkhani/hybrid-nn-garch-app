# Hybrid GARCH–Neural Network Models for Volatility Forecasting

Replication code for the Master's thesis:

> **A Systematic Formalization of Hybrid GARCH–Neural Network Approaches for Volatility Forecasting**  
> Humboldt-Universität zu Berlin

## Abstract

Volatility forecasting is central to financial econometrics, particularly for pricing and risk management. This thesis develops a systematic formalization of hybrid GARCH–neural network approaches, classifying them into three categories — **Augmented**, **Sequential**, and **Integrated** — and provides explicit mathematical formulations enabling consistent implementation and comparison. The frameworks are evaluated in a Monte Carlo simulation study based on EGARCH-type processes and in an empirical application to financial market data using squared returns and realized variance.

## Repository Structure

```
.
├── notebooks/
│   ├── 01_utilities.ipynb                  # Shared helper functions and preprocessing
│   ├── 02_augmented_architecture.ipynb     # Augmented hybrid models (Donaldson, MLP, LSTM)
│   ├── 03_integrated_architecture.ipynb    # Integrated hybrid models (MLP, LSTM)
│   ├── 04_simulation_study.ipynb           # Single-run EGARCH simulation across γ scenarios
│   ├── 05_monte_carlo.ipynb                # Full Monte Carlo evaluation (1000 runs)
│   ├── 06_data_application.ipynb           # Empirical application to financial market data
│   └── 07_evaluation_plots.ipynb           # Results aggregation and publication figures
├── requirements.txt
├── .gitignore
└── README.md
```

## Models Implemented

### Benchmark Models
| Model | Framework |
|---|---|
| GARCH(1,1) | `arch` library |
| EGARCH(1,1) | R `rugarch` via `rpy2` |

### Hybrid Architectures
| Category | Variant | Framework |
|---|---|---|
| Augmented | Donaldson–Kamstra (sigmoid basis) | PyTorch |
| Augmented | GARCH+MLP joint training | PyTorch |
| Augmented | GARCH+LSTM joint training | TensorFlow/Keras |
| Sequential | GARCH residuals → MLP | TensorFlow/Keras |
| Sequential | GARCH residuals → LSTM | TensorFlow/Keras |
| Integrated | IntegratedMLP | PyTorch |
| Integrated | IntegratedLSTM | PyTorch |

## Notebook Dependencies

The notebooks use Jupyter's `%run` magic to share code. Always run them from within the `notebooks/` directory, or open them there as the working directory. The dependency order is:

```
01_utilities.ipynb
    ↑ loaded by all other notebooks via %run

02_augmented_architecture.ipynb  ←┐
03_integrated_architecture.ipynb ←┤ loaded by 04, 05, 06
                                  │
04_simulation_study.ipynb         │ single-scenario EGARCH sim
05_monte_carlo.ipynb   ───────────┘ full 1000-run MC loop
06_data_application.ipynb           real financial data
07_evaluation_plots.ipynb           reads saved CSV results
```

## Setup

### Python environment

```bash
pip install -r requirements.txt
```

### R dependency (for EGARCH via rugarch)

```r
install.packages("rugarch")
```

The `rpy2` package bridges Python to R. Ensure R is installed and accessible from your PATH before running any EGARCH cells.

## Reproducibility

All stochastic components use fixed seeds (`seed = s` per Monte Carlo run). The simulation parameters are:

| Parameter | Value |
|---|---|
| Series length | n = 1500 |
| Train/test split | 80% / 20% |
| EGARCH ω | −0.3 |
| EGARCH α | 0.25 |
| EGARCH β | 0.90 |
| Leverage γ | {−0.25, −0.15, −0.08, 0.01, 0.25} |
| Monte Carlo runs | 1000 |
| Forecast horizon | h = 1, …, 100 |

## Forecast Evaluation

Models are compared using the **QLIKE loss**:

$$\text{QLIKE}(h) = \mathbb{E}\left[\log \hat{\sigma}^2_{t+h} + \frac{r_{t+h}^2}{\hat{\sigma}^2_{t+h}}\right]$$

as well as RMSE, MAE, and MAPE. Volatility proxies are squared returns and realized variance.
