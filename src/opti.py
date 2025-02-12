import optuna
import numpy as np
from src.kelly import optimize_kelly_pairs_portfolio
from src.cointegration import find_cointegrated_pairs

def objective(trial, tickers, start_date, end_date):
    """
    Objective function for Optuna optimization of pairs trading strategy.
    
    Args:
        trial: Optuna trial object
        tickers (list): List of stock tickers to consider
        start_date (str): Start date in 'YYYY-MM-DD' format 
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        float: Negative total Sharpe ratio (for minimization)
    """
    # Parameters to optimize
    p_value_threshold = trial.suggest_float('p_value_threshold', 0.001, 0.1, log=True)
    min_half_life = trial.suggest_int('min_half_life', 1, 30)
    volatility_lookback = trial.suggest_int('volatility_lookback', 10, 100)
    
    # Find cointegrated pairs using optimized p-value threshold
    pairs = find_cointegrated_pairs(tickers, start_date, end_date, p_value_threshold)
    
    if not pairs:
        return 0.0
        
    pairs_list = [(p[0], p[1], p[3]) for p in pairs]
    portfolio = optimize_kelly_pairs_portfolio(pairs_list, start_date, end_date)
    total_sharpe = sum(alloc['sharpe_ratio'] for alloc in portfolio.values())
    
    # Return negative since Optuna minimizes
    return -total_sharpe

def optimize_strategy(tickers, start_date, end_date, n_trials=100):
    """
    Run Bayesian optimization to find optimal parameters for pairs trading strategy.
    
    Args:
        tickers (list): List of stock tickers to consider
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        n_trials (int): Number of optimization trials
        
    Returns:
        dict: Best parameters and study statistics
    """
    study = optuna.create_study(direction='minimize')
    
    # Create partial function with fixed arguments
    objective_func = lambda trial: objective(trial, tickers, start_date, end_date)
    
    study.optimize(objective_func, n_trials=n_trials)
    
    return {
        'best_params': study.best_params,
        'best_value': -study.best_value,  # Convert back to positive Sharpe
        'best_trial': study.best_trial,
        'trials_dataframe': study.trials_dataframe(),
        'study': study
    }
