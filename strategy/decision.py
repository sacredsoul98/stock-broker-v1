import pandas as pd
from config import settings
from utils.logger import get_logger

logger = get_logger("strategy.decision")

def generate_signals(df):
    """
    Generates action signals based on model probability.
    Only LR probabilities are used for strict signal generation.
    LSTM is used contextually as complementary data.
    """
    logger.info("Generating trading signals...")
    df_signals = df.copy()
    
    # 1: BUY, -1: SELL, 0: HOLD
    signals = []
    for prob in df_signals['LR_Prob']:
        if prob > settings.BUY_PROB_THRESHOLD:
            signals.append(1)
        elif prob < settings.SELL_PROB_THRESHOLD:
            signals.append(-1)
        else:
            signals.append(0)
            
    df_signals['Signal'] = signals
    
    # Calculate expected change from LSTM for context
    if 'LSTM_Pred_Price' in df_signals.columns:
        df_signals['LSTM_Expected_Return'] = (df_signals['LSTM_Pred_Price'] - df_signals['Close']) / df_signals['Close']
    
    return df_signals
