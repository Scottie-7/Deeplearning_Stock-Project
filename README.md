# Deep Learning Stock Project: Data Pipeline, Preprocessing & FI-2010 LOB Benchmark

**Author:** Scottie Dunnavan | **Course:** CSCI 4425/5425 Introduction to Deep Learning

Limit Order Book (LOB) preprocessing, atomic Parquet persistence, spatial representation engineering (`sonar_norm`), post-hoc temperature scaling calibration, and 3-seed deep learning & tabular benchmark evaluation on the Nasdaq Nordic **FI-2010** dataset (`report_ps6.pdf`).

## Repository Layout
- `data/data.txt`: Dataset links and specifications (FI-2010, Databento ITCH, U.S. small/mid-cap live spec).
- `src/main.py` & `src/preprocessing.py`: Offline synthetic smoke-test pipeline (atomic Parquet writes, Lee-Ready classifier, 10-level OFI, AIR, triple-barrier labels, baseline evaluator; requires zero API keys).
- `src/fi2010/`: Stage 6 FI-2010 deep learning (`DeepLOB`, `LSTM-Attention`) and tabular (`XGBoost`, `Random Forest`, `MLP`, `Majority Class`) training, calibration, and evaluation suite executed on NMSU `simurgh` (NVIDIA H100 NVL).
- `logs/fi2010/perstock_all_models.log`: Verbatim 3-seed (`Seeds 42, 43, 44`) training and evaluation log under the per-stock 80/20 split (`100`-row purge per stock) and boundary-masked windows (`T=100`).
- `requirements.txt`: Pinned core Python dependencies for offline preprocessing and evaluation.

## Stage 6 FI-2010 3-Seed Benchmark (`T=100`, `138,102` Clean Test Windows, `ddof=1`)

| Model (`Seeds 42, 43, 44`) | Input Representation | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Temp ($T$) | ECE (Raw -> Cal) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (`2000` cap, ES=30)** | `sonar_norm` 160f (lags 10/50/99) | `0.7601 +/- 0.0010` | **`0.7995 +/- 0.0010`** | **`0.8002 +/- 0.0011`** | `1.04 +/- 0.01` | `0.0612 -> 0.0687` |
| **Random Forest (`150` trees)** | `sonar_norm` 160f (lags 10/50/99) | `0.7418 +/- 0.0006` | `0.7870 +/- 0.0001` | `0.7892 +/- 0.0002` | `0.75 +/- 0.00` | `0.1437 -> 0.0848` |
| **DeepLOB (`decpre` raw, 25 ep)** | Raw FI-2010 `(1, 100, 40)` | `0.7135 +/- 0.0163` | `0.7349 +/- 0.0176` | `0.7353 +/- 0.0184` | `1.47 +/- 0.25` | `0.0649 -> 0.0151` |
| **DeepLOB (`sonar_norm`, 25 ep)** | `sonar_norm` `(1, 100, 40)` | `0.7195 +/- 0.0044` | `0.7330 +/- 0.0033` | `0.7319 +/- 0.0033` | `1.43 +/- 0.18` | `0.0689 -> 0.0114` |
| **LSTM-Attention v3 (25 ep)** | `sonar_norm` `(100, 40)` unscaled | `0.6615 +/- 0.0022` | `0.6688 +/- 0.0022` | `0.6683 +/- 0.0028` | `1.57 +/- 0.17` | `0.1015 -> 0.0078` |
| **MLP (`128, 64`, standardized)** | `sonar_norm` 160f (lags 10/50/99) | `0.6446 +/- 0.0078` | `0.6617 +/- 0.0090` | `0.6609 +/- 0.0089` | `1.86 +/- 0.15` | `0.1056 -> 0.0311` |
| **XGBoost Ablation (no lags)** | `sonar_norm` 40f snapshot only | `0.4036 +/- 0.0027` | `0.3463 +/- 0.0013` | `0.4140 +/- 0.0008` | `1.13 +/- 0.02` | `0.1152 -> 0.0955` |
| **Majority Class Baseline** | Prior (`Up = 0`) | — | `0.1699` | `0.3421` | — | — |

## Setup & Execution
1. **Install:** `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. **Run Offline Synthetic Smoke Test (No API Keys):** `python src/main.py`
3. **Reproduce FI-2010 Per-Stock Benchmark:** `CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python -u src/fi2010/run_perstock_clean.py`
