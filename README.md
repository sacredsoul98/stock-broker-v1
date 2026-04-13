# 📈 NIFTY 50 AI-Assisted Trading System

An end-to-end, modular, algorithmic trading framework built for generating statistical trading signals on the Indian index (**NIFTY 50**). 

The system leverages dual machine learning engines—a **PyTorch LSTM** neural network for continuous price forecasting, paired with a **Logistic Regression** classifier for discrete directional probability. It features a complete pipeline covering data ingestion, feature engineering, predictive modeling, and a risk-managed backtesting simulation.

> **Disclaimer:** This project is strictly designed for **paper trading and educational research**. It does not integrate with any real brokerage. Please DO NOT trade live capital blindly based on these unverified models without conducting significant out-of-sample forward testing.

---

## ⚡ Features

* 📊 **Automated Data Pipeline:** Pulls multi-year historical data from Yahoo Finance (`yfinance`).
* 🧠 **Dual-Model Inference:** 
  * **Logistic Regression:** Classifies market direction probabilities utilizing class weights and L2 regularization.
  * **Deep Learning LSTM:** A 3-layer recurrent neural network predicting exact forward-looking price scales.
* 🛠️ **Deep Feature Engineering:** Generates structural alpha indicators including MACD, Stochastic Oscillators, Volatility bands, and Moving Averages via the `ta` library.
* 🛡️ **Risk Management Backtesting Engine:** Simulates capital tracking incorporating hard Position Sizing (e.g. 20%), Stop-Losses, and Take-Profit bounds.
* 🐳 **Containerized Environment:** Wraps the entire orchestration cleanly inside a reproducible Podman/Docker ecosystem.

---

## 📁 Repository Structure

```text
nifty-ai-trading/
├── Containerfile            # Podman/Docker image definition
├── run_podman.sh            # One-click execution script
├── requirements.txt         # Core dependencies
├── main.py                  # CLI Orchestrator pipeline
├── config/                  
│   └── settings.py          # Centralized configuration (Capital, Parameters, Thresholds)
├── data/                    
│   └── fetch.py             # Yfinance fetching strategies
├── features/                
│   └── engineer.py          # Technical indicators & sequence formatting
├── models/                  
│   ├── train_lr.py          # Logistic Regression modeling logic
│   └── train_lstm.py        # PyTorch sequence modeling logic
├── strategy/                
│   └── decision.py          # Alpha generator / conditional mapping
└── backtesting/             
    └── engine.py            # Simulated PnL engine & equity tracking
```

---

## 🚀 Getting Started

### Prerequisites
You need **Podman** (or Docker) installed on your system to run the containerized application. 
No need to deal with local python environments or missing dependencies; the container handles everything.

### 1. Build and Run Entire Pipeline
The orchestrator natively updates your data, trains the ML/DL weights side-by-side, and simulates an environment backtest in one move.
```bash
# Make the run script executable
chmod +x run_podman.sh

# Complete run
./run_podman.sh --run
```

### 2. Manual Commands
Run modular layers independently using the CLI interface:
```bash
# Download/Update latest datasets only
./run_podman.sh --update

# Retrain both Neural Networks and Regression weights locally
./run_podman.sh --train

# Run backtest tracking and export metrics
./run_podman.sh --backtest
```

---

## 🔬 Outputs & Verification

Upon running a backtest, the system generates artifacts that land explicitly inside your local `output/` directory (mapped from the container):

- `predictions_comparison.csv` — A ledger tracking standard Logistic percentage confidences against actual modeled LSTM closing trajectories.
- `trades.csv` — An analytical record mapping every simulated Entry/Exit, triggered stops, and un-realized cash.
- `equity_curve.png` — Visual map of simulated rolling portfolio growth.

---

## ⚙️ Configuration (Tuning)
You can easily tune hyper-parameters and adjust risk controls by modifying values within the centralized `config/settings.py` file:
- `INITIAL_CAPITAL`
- `STOP_LOSS_PCT` & `TAKE_PROFIT_PCT`
- `LSTM_EPOCHS`, `LSTM_HIDDEN_DIM`, `LSTM_NUM_LAYERS`
- `BUY_PROB_THRESHOLD`

*Happy Quanting!* 🚀
