# Deep Learning Stock Project - Data Pipeline (`src/`)

This directory contains the data collection engine and preprocessing pipelines for limit order book (LOB) deep learning models.

## Repository Layout
- `data/data.txt`: Dataset download links (FI-2010, Databento, self-recorded).
- `src/`: Python source code for data streaming, collection, and preprocessing.
- `report_ps3.pdf`: Stage 3 project report.

## Setup & Execution

### 1. Environment & Dependencies
Python 3.10+ required.

```bash
# Clone repository
git clone [https://github.com/Scottie-7/Deeplearning_Stock-Project.git](https://github.com/Scottie-7/Deeplearning_Stock-Project.git)
cd Deeplearning_Stock-Project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy aiohttp websockets python-dotenv
```

### 2. Configuration (`.env`)
Create a `.env` file in the project root:

```env
TASTYTRADE_USER=your_username
TASTYTRADE_PASS=your_password
WEBULL_TOKEN=your_token
MOOMOO_HOST=127.0.0.1
MOOMOO_PORT=11111
```

### 3. Execution
To run the market data collection engine:

```bash
python src/main.py
```
