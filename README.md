# Deep Learning Stock Project - Data Pipeline & Preprocessing Engine
**Author:** Scottie Dunnavan  
**Course:** CSCI 4425/5425 Introduction to Deep Learning  

This repository contains the dataset specifications, ingestion framework, and preprocessing engine for Limit Order Book (LOB) deep learning models. This serves as the core data and modeling component of the Sonar Stock Project (see `report_ps4.pdf` in the repo root for full context).

## Repository Layout
- `data/data.txt`: Dataset download links (FI-2010, Databento, self-recorded).
- `src/`: Python source code for data streaming, crash-safe Parquet recording, microstructure feature extraction, and Triple-Barrier labeling.
- `report_ps4.pdf`: Stage 4 project report.

## Market Data Engine & Preprocessing (`src/`)
### Status
Active development (Stage 4 deliverable). The repository includes both our foundational live Tastytrade L1 polling connector and our newly implemented v1 Preprocessing & Baseline Solution.

### File Structure
| File | Purpose |
| :--- | :--- |
| `src/main.py` | Dual-mode entry point. Defaults to an out-of-the-box offline verification pipeline for grading/testing; supports `--live` flag for active streaming. |
| `src/preprocessing.py` | Core v1 preprocessing engine: crash-safe atomic Parquet writer
