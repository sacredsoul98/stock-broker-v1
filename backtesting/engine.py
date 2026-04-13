import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from config import settings
from utils.logger import get_logger

logger = get_logger("backtesting.engine")

class BacktestEngine:
    def __init__(self, data):
        self.data = data
        self.capital = settings.INITIAL_CAPITAL
        self.position_size_pct = settings.POSITION_SIZE_PCT
        self.stop_loss = settings.STOP_LOSS_PCT
        self.take_profit = settings.TAKE_PROFIT_PCT
        self.trades_log = []
        
        # State variables
        self.position = 0 # 0 for flat, 1 for long, -1 for short
        self.entry_price = 0.0
        self.entry_date = None
        self.position_size = 0.0 # monetary value allocation
        self.quantity = 0 # number of shares/units
        
        # Tracking
        self.equity_curve = [self.capital]
        self.equity_dates = []

    def close_position(self, current_date, current_price, reason):
        if self.position == 1: # Close Long
            pnl = (current_price - self.entry_price) * self.quantity
            self.capital += self.position_size + pnl
            self.trades_log.append({
                'Entry_Date': self.entry_date,
                'Exit_Date': current_date,
                'Type': 'LONG',
                'Entry_Price': self.entry_price,
                'Exit_Price': current_price,
                'PnL': pnl,
                'Reason': reason
            })
            
        elif self.position == -1: # Close Short
            pnl = (self.entry_price - current_price) * self.quantity
            self.capital += self.position_size + pnl
            self.trades_log.append({
                'Entry_Date': self.entry_date,
                'Exit_Date': current_date,
                'Type': 'SHORT',
                'Entry_Price': self.entry_price,
                'Exit_Price': current_price,
                'PnL': pnl,
                'Reason': reason
            })
            
        self.position = 0
        self.position_size = 0.0
        self.quantity = 0

    def open_position(self, current_date, current_price, direction):
        if self.position != 0:
            return # Only one open position at a time
            
        allocation = self.capital * self.position_size_pct
        self.quantity = allocation / current_price
        self.position_size = allocation
        self.capital -= allocation # Lock capital in position
        
        self.position = direction # 1 for LONG, -1 for SHORT
        self.entry_price = current_price
        self.entry_date = current_date

    def run(self):
        logger.info("Starting Backtest Simulation...")
        for date, row in self.data.iterrows():
            close_price = row['Close']
            signal = row['Signal']
            self.equity_dates.append(date)
            
            # 1. Check if we need to close existing position (Risk Management)
            if self.position != 0:
                if self.position == 1:
                    price_change = (close_price - self.entry_price) / self.entry_price
                    if price_change <= -self.stop_loss:
                        self.close_position(date, close_price, 'Stop Loss')
                    elif price_change >= self.take_profit:
                        self.close_position(date, close_price, 'Take Profit')
                        
                elif self.position == -1:
                    price_change = (self.entry_price - close_price) / self.entry_price
                    if price_change <= -self.stop_loss:
                        self.close_position(date, close_price, 'Stop Loss')
                    elif price_change >= self.take_profit:
                        self.close_position(date, close_price, 'Take Profit')

            # 2. Check if we should close position due to opposite signal
            if self.position == 1 and signal == -1:
                self.close_position(date, close_price, 'Reversal Signal')
            elif self.position == -1 and signal == 1:
                self.close_position(date, close_price, 'Reversal Signal')

            # 3. Open new position if flat
            if self.position == 0:
                if signal == 1:
                    self.open_position(date, close_price, 1)
                elif signal == -1:
                    self.open_position(date, close_price, -1)
                    
            # Update equity tracking (Cash + Unrealized PnL)
            current_equity = self.capital
            if self.position == 1:
                current_equity += self.quantity * close_price
            elif self.position == -1:
                current_equity += self.position_size + (self.entry_price - close_price) * self.quantity
                
            self.equity_curve.append(current_equity)
            
        # Close any open positions at the end of the simulation
        if self.position != 0:
            last_date = self.data.index[-1]
            last_price = self.data.iloc[-1]['Close']
            self.close_position(last_date, last_price, 'End of Backtest')

        self.equity_curve = self.equity_curve[1:] # Align lengths
        logger.info("Backtest Simulation Completed.")
        
    def get_metrics(self):
        trades_df = pd.DataFrame(self.trades_log)
        initial_cap = settings.INITIAL_CAPITAL
        final_cap = self.capital
        return_pct = ((final_cap - initial_cap) / initial_cap) * 100
        
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['PnL'] > 0]) if total_trades > 0 else 0
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Max Drawdown
        equity_s = pd.Series(self.equity_curve)
        rolling_max = equity_s.cummax()
        drawdown = (equity_s - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        logger.info("--- Backtest Metrics ---")
        logger.info(f"Total Return: {return_pct:.2f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate:     {win_rate:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Final Capital: ₹{final_cap:.2f}")
        
        return trades_df
        
    def save_results(self, output_dir=os.path.join(settings.BASE_DIR, "output")):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save side-by-side predictions
        preds_path = os.path.join(output_dir, "predictions_comparison.csv")
        # Ensure datetimes are index properly when saving
        self.data.to_csv(preds_path)
        
        # Save Trades Log
        trades_df = pd.DataFrame(self.trades_log)
        trades_path = os.path.join(output_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False)
        
        # Plot Equity Curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.equity_dates, self.equity_curve, label='Portfolio Value')
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital (₹)')
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(output_dir, "equity_curve.png")
        plt.savefig(plot_path)
        logger.info(f"Results saved to {output_dir}")
