import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data fetching settings
TICKER = "^NSEI"  # NIFTY 50 Index
DATA_FILE = os.path.join(DATA_DIR, "nifty50_data.csv")
HISTORICAL_PERIOD = "10y"  # 5 years of historical data

# Feature engineering parameters
SMA_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
VOLATILITY_WINDOW = 20

# LSTM parameters
LSTM_SEQ_LENGTH = 100
LSTM_EPOCHS = 200
LSTM_BATCH_SIZE = 64
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 4
LSTM_DROPOUT = 0.2

# Trading Logic parameters
BUY_PROB_THRESHOLD = 0.6
SELL_PROB_THRESHOLD = 0.4

# Risk Management
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.20  # 20% of current capital per trade
STOP_LOSS_PCT = 0.015     # 1.5% stop loss
TAKE_PROFIT_PCT = 0.03    # 3.0% take profit
