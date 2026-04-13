import argparse
import os
import pandas as pd
from config import settings
from utils.logger import get_logger

from data.fetch import fetch_historical_data, update_data, load_data
from features.engineer import create_features, create_lstm_sequences
from models.train_lr import train_logistic_regression
from models.train_lstm import train_lstm_model
from strategy.decision import generate_signals
from backtesting.engine import BacktestEngine

logger = get_logger("main")

def main():
    parser = argparse.ArgumentParser(description="AI-Assisted Index Trading System")
    parser.add_argument('--fetch', action='store_true', help="Fetch historical data from scratch")
    parser.add_argument('--update', action='store_true', help="Update existing data with latest prices")
    parser.add_argument('--train', action='store_true', help="Train ML and DL Models (LR and LSTM)")
    parser.add_argument('--backtest', action='store_true', help="Run the backtest engine on the test set")
    parser.add_argument('--run', action='store_true', help="Run full pipeline: update, train, backtest")
    
    args = parser.parse_args()
    
    if args.run:
        args.update = True
        args.train = True
        args.backtest = True

    # 1. DATA PHASE
    if args.fetch:
        fetch_historical_data()
    elif args.update:
        update_data()
        
    # Validation step
    if not os.path.exists(settings.DATA_FILE):
        logger.error(f"Data file not found at {settings.DATA_FILE}. Please run with --fetch first.")
        return

    # 2. TRAINING PHASE
    if args.train:
        logger.info("Starting Training Pipeline...")
        df = load_data()
        if len(df) < 200:
            logger.error("Not enough data to train models efficiently.")
            return
            
        # Feature Engineering for LR
        engineered_df = create_features(df)
        
        # Train Logistic Regression
        lr_results, _, _ = train_logistic_regression(engineered_df)
        
        # Feature Engineering for LSTM
        X_seq, y_val, _ = create_lstm_sequences(engineered_df)
        lstm_results, _, _ = train_lstm_model(engineered_df, X_seq, y_val)
        
        # We save results so we can load them during backtesting locally without retraining if we want
        # In a real environment, we would infer these live.
        results = pd.concat([lr_results, lstm_results], axis=1)
        # Add actual close prices back onto results
        results = results.join(engineered_df['Close'], how='inner')
        results.to_csv(os.path.join(settings.BASE_DIR, "output", "test_predictions.csv"))
        logger.info("Training complete and test predictions saved.")
            
    # 3. BACKTEST PHASE
    if args.backtest:
        logger.info("Starting Backtesting Pipeline...")
        preds_path = os.path.join(settings.BASE_DIR, "output", "test_predictions.csv")
        if not os.path.exists(preds_path):
            logger.error(f"Predictions not found at {preds_path}. Please run with --train first.")
            return
            
        # Load the predictions we generated on the test set
        test_df = pd.read_csv(preds_path, index_col=0, parse_dates=True)
        
        # Strategy logic
        signals_df = generate_signals(test_df)
        
        # Backtest
        engine = BacktestEngine(signals_df)
        engine.run()
        engine.get_metrics()
        engine.save_results()
        
if __name__ == "__main__":
    main()
