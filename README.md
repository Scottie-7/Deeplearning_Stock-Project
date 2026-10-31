# Deep Learning Stock Project: Data Pipeline, Preprocessing & Baseline Engine

**Author:** Scottie Dunnavan
**Course:** CSCI 4425/5425 Introduction to Deep Learning

Dataset specs, live broker adapters, crash-safe Parquet persistence, and LOB preprocessing and feature pipeline for the Stock Project. Architecture context: `report_ps5.pdf` (Stage 5), `report_ps4.pdf` (Stage 4), `report_ps3.pdf` (Stage 3).

## Repository Layout

- `data/data.txt`: dataset links and specs (FI-2010, Databento ITCH pull, self-recorded U.S. small-cap live data).
- `src/`: preprocessing, Parquet recording, features, labeling, baselines, live streaming.
- `report_ps5.pdf`: Stage 5 report. `report_ps4.pdf`, `report_ps3.pdf`: earlier stages.

## Current Status: Stage 5 (`ps5-sol2`)

Implemented:
- Atomic `.parquet` writes (temp file, then `os.replace()`) with zstd. `l1_quotes` (18 columns) and wide `l2_snapshots` (47 columns, 10 levels, `session`, `book_age_ms`).
- Lee-Ready trade side (+1 / -1 / 0) with tick-test fallback.
- Features: relative price distance, 10-level price-keyed OFI, AIR (single-venue definition and threshold flags pending), micro-price divergence.
- Triple-barrier labels (Down=0, Neutral=1, Up=2), 50-tick horizon, 15 bps minimum barrier.
- Tensors: spatial (Batch, 1, 100, 40), auxiliary flow (Batch, 100, 16), tabular (Batch, 56).
- Chronological 70/30 split with a 50-sample purge.
- Models: majority-class, Random Forest (150 trees, balanced), MLP (128, 64). Metrics: macro-F1, per-class F1, 3x3 confusion matrix.

Not implemented: LSTM-Attention and DeepLOB (PyTorch), temperature scaling and ECE, locked v1.2 shapes (Batch, 63), (Batch, 60, 10), (Batch, 1, 600, 40) with (Batch, 600, 20), venue-matched AIR and its flags (ICEBERG_ABSORPTION, UNEXECUTED_DEPLETION), 5-minute buffered flush and bounded queue. Full list: report Section 6.

## File Structure (`src/`)

| File | Purpose |
| :--- | :--- |
| `src/main.py` | Entry point. Runs the offline pipeline by default, no API keys. `--live` starts Tastytrade streaming. |
| `src/preprocessing.py` | Parquet writer, Lee-Ready classifier, feature calculators, triple-barrier labeler, chronological evaluator. |
| `src/providers/__init__.py` | Provider package initializer. |
| `src/providers/base.py` | `MarketDataSource` abstract interface (`connect`, `stream_data`). |
| `src/providers/tastytrade.py` | Tastytrade OAuth2 and live L1 adapter. |

## Setup & Execution

### 1. Install

    cd src
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
    pip install numpy pyarrow scikit-learn

Live mode only: `pip install aiohttp websockets python-dotenv rich`

`preprocessing.py` has pure-NumPy fallbacks if `pyarrow` or `scikit-learn` is missing.

### 2. Run (default, no `.env` or API keys)

    cd src
    python main.py

Writes Parquet partitions to `src/recordings/` and prints the model comparison table and confusion matrices (Majority, Random Forest, MLP). Data is synthetic. The generator ties order-book imbalance and trade side to the next price move, so scores verify the pipeline runs end to end, not market predictability. See report Section 5.

### 3. Live Tastytrade L1 Streaming (optional)

Brokerage sessions enforce single-active-session or single-device limits, and live recording runs across premarket, regular, and after-hours sessions. Sharing `.env` credentials during grading could disconnect the recorder or invalidate its token. Default execution is therefore isolated from broker sessions.

Create a git-ignored `src/.env`:

    TASTY_CLIENT_ID=your_client_id
    TASTY_CLIENT_SECRET=your_client_secret
    TASTY_REFRESH_TOKEN=your_refresh_token
    TASTY_ENV=production

Never commit `.env`, client secrets, or refresh tokens. Start streaming:

    python main.py --live

## Known Issues

- Default barrier multiplier is `k_vol=0.45`. The locked spec uses initial `k=1.0`, fit on the train split. Align before V2.
- A level that vanishes without scrolling out of the top 10 is assigned to the nearest current level (OFI approximation).
- Stage 5 code has not been executed end to end. Confirm `python main.py` runs before relying on this README's claims.

## Next Milestones (Stage 1A / 1B Roadmap)

- **Multi-venue recorder and tier engine:** Moomoo OpenD (NYSE ArcaBook L2) and IBKR `ib_async` (Nasdaq TotalView L2) under the 600-second minimum L2 dwell lock and dual-pool slot router.
- **NMSU `simurgh` H100 runs:** `RF/XGBoost`, `LSTM-Attention` (60 one-minute bars), and `DeepLOB` (100 ms L2 lookback, FI-2010 transfer learning, 0.1x LR on unfrozen CNN) on the shared 180 s horizon.
- **Sonar live dashboard:** live streams, AIR absorption highlights, and model probability gauges in the local web UI.
