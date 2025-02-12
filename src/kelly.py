import numpy as np
import pandas as pd
from src.market_data_fetcher import get_stock_data

def calculate_kelly_position_sizes(ticker1, ticker2, hedge_ratio, start_date, end_date, 
                                 lookback_window=252, price_type='Close'):
    """
    Calculate Kelly Criterion based position sizes for a pair of cointegrated assets.
    
    Args:
        ticker1 (str): First stock symbol
        ticker2 (str): Second stock symbol
        hedge_ratio (float): Hedge ratio between the two assets
        start_date (str): Start date in 'YYYY-MM-DD' format 
        end_date (str): End date in 'YYYY-MM-DD' format
        lookback_window (int): Number of days to use for calculating statistics
        price_type (str): Type of price to use
        
    Returns:
        dict: Dictionary containing Kelly position sizes and statistics
    """
    # Get price data
    df1 = get_stock_data(ticker1, start_date, end_date)
    df2 = get_stock_data(ticker2, start_date, end_date)
    
    # Calculate spread
    spread = df1[price_type] - hedge_ratio * df2[price_type]
    
    # Calculate spread returns
    spread_returns = spread.pct_change().dropna()
    
    # Calculate mean and standard deviation of returns
    mean_return = spread_returns.mean()
    std_return = spread_returns.std()
    
    # Calculate win probability and win/loss ratio based on historical data
    wins = spread_returns[spread_returns > 0]
    losses = spread_returns[spread_returns < 0]
    zero_returns = spread_returns[spread_returns == 0]
    
    total_trades = len(spread_returns)
    if total_trades == 0:
        return {
            'kelly_fraction': 0,
            'position_size_1': 0,
            'position_size_2': 0,
            'win_probability': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'mean_return': 0,
            'std_return': 0,
            'sharpe_ratio': 0
        }
    
    # Count zero returns as half wins
    win_prob = (len(wins) + len(zero_returns) * 0.5) / total_trades
    
    # Handle cases where there are no wins or losses
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1  # Use 1 to avoid division by zero
    
    # Calculate Kelly fraction
    kelly_fraction = (win_prob / avg_loss) - ((1 - win_prob) / avg_win)
    
    # Apply a safety factor of 0.5 to be more conservative
    kelly_fraction *= 0.5
    
    # Calculate position sizes
    notional_exposure = 1.0  # Normalized to 1 unit of capital
    position1_size = kelly_fraction * notional_exposure
    position2_size = -kelly_fraction * notional_exposure * hedge_ratio
    
    return {
        'kelly_fraction': kelly_fraction,
        'position_size_1': position1_size,
        'position_size_2': position2_size,
        'win_probability': win_prob,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': mean_return / std_return if std_return != 0 else 0
    }

def optimize_kelly_pairs_portfolio(pairs_list, start_date, end_date):
    """
    Optimize Kelly position sizes across multiple pairs.
    
    Args:
        pairs_list (list): List of tuples containing (ticker1, ticker2, hedge_ratio)
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: Dictionary containing optimized position sizes for each pair
    """
    portfolio_allocations = {}
    
    # Calculate individual Kelly sizes for each pair
    for ticker1, ticker2, hedge_ratio in pairs_list:
        kelly_stats = calculate_kelly_position_sizes(
            ticker1, ticker2, hedge_ratio, start_date, end_date
        )
        
        # Store results
        pair_key = f"{ticker1}-{ticker2}"
        portfolio_allocations[pair_key] = {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'position_size_1': kelly_stats['position_size_1'],
            'position_size_2': kelly_stats['position_size_2'],
            'sharpe_ratio': kelly_stats['sharpe_ratio'],
            'kelly_fraction': kelly_stats['kelly_fraction']
        }
    
    total_sharpe = sum(alloc['sharpe_ratio'] for alloc in portfolio_allocations.values())
    
    if total_sharpe > 0:
        for pair in portfolio_allocations:
            weight = portfolio_allocations[pair]['sharpe_ratio'] / total_sharpe
            portfolio_allocations[pair]['position_size_1'] *= weight
            portfolio_allocations[pair]['position_size_2'] *= weight
            
    return portfolio_allocations
