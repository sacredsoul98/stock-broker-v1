import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from config import settings
from utils.logger import get_logger

logger = get_logger("data.fetch")

def fetch_historical_data(ticker=settings.TICKER, period=settings.HISTORICAL_PERIOD, filepath=settings.DATA_FILE):
    """
    Fetches historical stock data using yfinance and saves to CSV.
    """
    logger.info(f"Fetching {period} historical data for {ticker}...")
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            logger.error("No data fetched. Check the ticker symbol or your network connection.")
            return False
            
        # yfinance columns might come as MultiIndex, flatten if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data.reset_index(inplace=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Successfully saved data to {filepath} with shape {data.shape}.")
        return True
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return False

def update_data(ticker=settings.TICKER, filepath=settings.DATA_FILE):
    """
    Updates the existing CSV file with missing newer data.
    """
    if not os.path.exists(filepath):
        logger.info("Data file not found. Fetching historical data from scratch.")
        return fetch_historical_data(ticker=ticker, filepath=filepath)
        
    try:
        existing_data = pd.read_csv(filepath, parse_dates=['Date'])
        if existing_data.empty:
            return fetch_historical_data(ticker=ticker, filepath=filepath)
            
        last_date = existing_data['Date'].max()
        today_str = datetime.now().strftime('%Y-%m-%d')
        last_date_str = last_date.strftime('%Y-%m-%d')
        
        if last_date_str >= today_str:
            logger.info("Data is already up to date.")
            return True
            
        logger.info(f"Updating data from {last_date_str} to present...")
        
        # Start fetching from the day after the last date
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        new_data = yf.download(ticker, start=start_date)
        
        if new_data.empty:
            logger.info("No new data available.")
            return True
        
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] for col in new_data.columns]
            
        new_data.reset_index(inplace=True)
        
        # Ensure date format consistency
        new_data['Date'] = pd.to_datetime(new_data['Date'])
        
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['Date'], keep='last')
        combined_data.sort_values(by='Date', inplace=True)
        combined_data.to_csv(filepath, index=False)
        
        logger.info(f"Appended {len(new_data)} new rows. Total records: {len(combined_data)}.")
        return True
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return False
        
def load_data(filepath=settings.DATA_FILE):
    """
    Loads data from the CSV file.
    """
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist. Call fetch_historical_data first.")
        return pd.DataFrame()
        
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df
