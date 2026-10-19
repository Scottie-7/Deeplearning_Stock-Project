# Deep Learning Stock Project — Data Pipeline, Preprocessing & Baseline Engine

**Author:** Scottie Dunnavan  
**Course:** CSCI 4425/5425 Introduction to Deep Learning  

This repository contains the dataset specifications, live broker ingestion adapters, crash-safe Parquet persistence layer, and Limit Order Book (LOB) preprocessing and feature-engineering pipeline for the Stock Project. For full architectural context, see `report_ps4.pdf` and `report_ps3.pdf` in the repository root.

## Repository Layout

- `data/data.txt`: Dataset download links and specifications, including the FI-2010 benchmark, Databento historical ITCH data pull, and self-recorded live U.S. small-cap data.
- `src/`: Runnable Python source code for LOB data preprocessing, atomic Parquet recording, microstructure feature extraction, Triple-Barrier labeling, preliminary baseline evaluation, and live broker streaming.
- `report_ps4.pdf`: Stage 4 project report covering the updated problem formulation, refined wide Parquet schemas, preprocessing implementation, and preliminary baseline solution.
- `report_ps3.pdf`: Stage 3 dataset description report.

## Market Data & Preprocessing Engine (`src/`)

### Current Status: Stage 4 (`ps4-sol1`)

1. **Runnable Preprocessing, Parquet Persistence & Preliminary Baseline (`src/preprocessing.py` and `src/main.py`):**
   - Fully functional out of the box, with **zero external API keys required**.
   - Crash-safe atomic `.parquet` writes for `l1_quotes` and 45-column flat, wide `l2_snapshots`.
   - Deterministic Lee-Ready `trade_side` aggressor classification (`+1` / `-1`).
   - 100 ms software grid alignment.
   - Microstructure feature extraction:
     - Relative Price Distance (`RelPriceDist`).
     - 10-level Multi-Level Order Flow Imbalance (`MLOFI`).
     - Absorption & Iceberg Ratio (`AIR` — Absorption vs. Spoof ratio).
     - Micro-Price Divergence.
   - Spread-Adjusted Triple-Barrier labeling (`Down = 0`, `Neutral = 1`, `Up = 2`).
   - Sliding-window tensor construction:
     - DeepLOB 4D Tensor: `(Batch, 1, 100, 40)`.
     - Tabular Baselines 2D Matrix: `(Batch, 46)`.
   - Chronological Random Forest baseline evaluation, reporting Macro-F1 and a 3x3 confusion matrix.

2. **Live Broker Ingestion (`src/providers/`):**
   - Tastytrade OAuth2 Level 1 (L1) streaming, implemented in `tastytrade.py`, is available through the `--live` flag when a local `.env` file is configured.

### File Structure (`src/`)

| File | Purpose |
| :--- | :--- |
| `src/main.py` | Entry point. Runs the self-contained Stage 4 preprocessing, Parquet persistence, tensor construction, and baseline evaluation pipeline by default, without requiring API keys. Supports `--live` for live Tastytrade streaming. |
| `src/preprocessing.py` | Stage 4 core solution: atomic `.tmp` to `os.replace()` Parquet writer using Zstandard (`zstd`), Lee-Ready quote-and-tick classifier, `RelPriceDist` / `MLOFI` / `AIR` / Micro-Price feature calculators, Spread-Adjusted Triple-Barrier labeler, and chronological baseline evaluator. |
| `src/providers/__init__.py` | Provider package initializer. |
| `src/providers/base.py` | `MarketDataSource` abstract base interface, defining `connect` and `stream_data` methods implemented by broker adapters. |
| `src/providers/tastytrade.py` | Tastytrade OAuth2 integration and live L1 quote-polling adapter. |

## Setup & Execution

### 1. Install Dependencies
From the repository root, navigate to `src/`, create a virtual environment, activate it, and install the dependencies:

    cd src
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
    pip install numpy pyarrow scikit-learn aiohttp websockets python-dotenv rich

*Note:* `preprocessing.py` also includes pure-NumPy fallbacks if `pyarrow` or `scikit-learn` are unavailable in a minimal grading environment.

### 2. Run Stage 4 Preprocessing & Preliminary Baseline (Default — No `.env` or API Keys Required)
From the `src/` directory, run:

    python main.py

This runs the end-to-end verification pipeline, writes atomic `.parquet` partitions under `src/recordings/`, constructs the 4D DeepLOB and 2D baseline feature arrays, fits the chronological baseline model, and prints the 8-step verification report to the terminal, including Macro-F1 and the 3x3 confusion matrix.

### 3. Run Live Tastytrade L1 Streaming (Optional — Requires Local `.env`)

**Why Live API Sessions Are Isolated from Default Execution:**  
Brokerage APIs and local gateways—including Interactive Brokers TWS/IB Gateway, Moomoo OpenD, Webull, and Tastytrade OAuth grants—enforce single-active-session or single-device restrictions and maintain token-refresh state. Because the live data recorder is designed to operate continuously across premarket, regular, and after-hours sessions (4:00 AM–8:00 PM ET) as well as overnight windows, sharing live `.env` credentials or triggering concurrent external logins during grading could cause session-conflict disconnects or invalidate active recording tokens. Therefore, default execution (`python main.py`) is strictly isolated from external brokerage sessions and runs the self-contained verification pipeline.

To run the live Tastytrade streamer locally, create a Git-ignored `src/.env` file containing your own credentials:

    TASTY_CLIENT_ID=your_client_id
    TASTY_CLIENT_SECRET=your_client_secret
    TASTY_REFRESH_TOKEN=your_refresh_token
    TASTY_ENV=production

*Security note:* Never commit your `.env` file, API credentials, client secrets, or refresh tokens to GitHub.

To start live streaming from the `src/` directory, run:

    python main.py --live

## Next Milestones (Stage 1A / 1B Roadmap)
- **Full Multi-Venue Recorder & Tier Engine:** Expand `providers/` with Moomoo `OpenD` (NYSE ArcaBook L2) and IBKR `ib_async` (Nasdaq TotalView L2) under the 600-second minimum L2 dwell lock and dual-pool slot router.
- **NMSU `simurgh` H100 Training Runs:** Execute full-scale `RF/XGBoost`, `LSTM/TFT` (60-minute 1m bar lookback), and `DeepLOB` (60-second 100ms L2 lookback with FI-2010 transfer learning and unfrozen 0.1x LR spatial CNNs) on the shared 3-minute (`180s`) horizon.
- **Sonar Live Visual Dashboard:** Connect live streams, DOM/Heatmap `AIR` absorption/spoof highlights, and competing 3-minute model probability gauges into the local web UI.
