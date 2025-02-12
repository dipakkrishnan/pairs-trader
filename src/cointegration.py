import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from src.market_data_fetcher import get_stock_data

def test_cointegration(ticker1, ticker2, start_date, end_date, price_type='Close'):
    """
    Test for cointegration between two assets using the Engle-Granger two-step method.
    
    Args:
        ticker1 (str): First stock symbol
        ticker2 (str): Second stock symbol 
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        price_type (str): Type of price to use ('Open', 'High', 'Low', 'Close', 'Adj Close')
        
    Returns:
        dict: Dictionary containing:
            - p_value: P-value from cointegration test
            - t_stat: Test statistic
            - critical_values: Critical values at 1%, 5%, and 10% levels
            - hedge_ratio: Hedge ratio between the two assets
    """
    # Fetch price data
    df1 = get_stock_data(ticker1, start_date, end_date)
    df2 = get_stock_data(ticker2, start_date, end_date)
    prices1 = df1[price_type]
    prices2 = df2[price_type]
    prices = pd.concat([prices1, prices2], axis=1).dropna()
    prices.columns = [ticker1, ticker2]
    
    # Perform cointegration test
    t_stat, p_value, critical_values = coint(prices[ticker1], prices[ticker2])
    
    # Calculate hedge ratio using linear regression
    hedge_ratio = np.polyfit(prices[ticker1], prices[ticker2], 1)[0]
    
    return {
        'p_value': p_value,
        't_stat': t_stat,
        'critical_values': critical_values,
        'hedge_ratio': hedge_ratio
    }

def find_cointegrated_pairs(tickers, start_date, end_date, p_value_threshold=0.05):
    """
    Find all cointegrated pairs within a list of tickers.
    
    Args:
        tickers (list): List of stock symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        p_value_threshold (float): Maximum p-value to consider pairs cointegrated
        
    Returns:
        list: List of tuples containing (ticker1, ticker2, p_value, hedge_ratio)
    """
    n = len(tickers)
    cointegrated_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                results = test_cointegration(tickers[i], tickers[j], start_date, end_date)
                
                if results['p_value'] < p_value_threshold:
                    cointegrated_pairs.append((
                        tickers[i],
                        tickers[j],
                        results['p_value'],
                        results['hedge_ratio']
                    ))
                    
            except Exception as e:
                print(f"Error testing pair {tickers[i]}-{tickers[j]}: {str(e)}")
                
    return cointegrated_pairs
