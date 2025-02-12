import yfinance as yf
import pandas as pd
import os

def get_stock_data(symbol, start_date, end_date, cache_dir='data_cache'):
    """
    Fetches stock data for a given symbol and date range, caches the data 
    in a specified directory, and retrieves it from the cache if available.
    
    Returns:
        pd.DataFrame: The stock data for the given symbol and date range.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{symbol}_{start_date}_{end_date}.parquet" 
    
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file, engine='pyarrow')
        
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_parquet(cache_file, engine='pyarrow')
    return data
