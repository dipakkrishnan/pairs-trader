import matplotlib.pyplot as plt
import pandas as pd
from src.market_data_fetcher import get_stock_data

def plot_asset_prices(tickers, start_date, end_date, price_type='Close'):
    """
    Plot asset prices for multiple tickers over a specified time period.
    
    Args:
        tickers (list): List of stock symbols to plot
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        price_type (str): Type of price to plot ('Open', 'High', 'Low', 'Close', 'Adj Close')
    """
    plt.figure(figsize=(14, 7))
    
    for ticker in tickers:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            
            if price_type in data.columns:
                # Normalize the prices to start at 100 for better comparison
                normalized_prices = data[price_type] / data[price_type].iloc[0] * 100
                plt.plot(data.index, normalized_prices, label=f'{ticker}')
            else:
                print(f"Warning: '{price_type}' column not found in data for {ticker}")
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    plt.title(f'Normalized Asset Prices Over Time ({price_type})')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Base=100)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# Example usage:
tickers = ['AAPL', 'MSFT', 'GOOGL']
plot_asset_prices(tickers, '2023-01-01', '2023-12-31')
plt.show()

