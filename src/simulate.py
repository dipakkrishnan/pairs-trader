
import pandas as pd
from src.opti import optimize_strategy
from src.cointegration import find_cointegrated_pairs
from src.kelly import optimize_kelly_pairs_portfolio

def run_pairs_trading_simulation(tickers, start_date, end_date, n_trials=100):
    """
    Run end-to-end pairs trading simulation including optimization and portfolio construction.
    
    Args:
        tickers (list): List of stock tickers to consider
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        n_trials (int): Number of optimization trials
        
    Returns:
        dict: Simulation results including optimal parameters and portfolio allocations
    """
    # Step 1: Run Bayesian optimization to find best parameters
    print("Running Bayesian optimization...")
    optimization_results = optimize_strategy(tickers, start_date, end_date, n_trials)
    best_params = optimization_results['best_params']
    
    print("\n" + "="*50)
    print("Optimization Results")
    print("="*50)
    print(f"Best Sharpe Ratio: {optimization_results['best_value']:.4f}")
    print("\nHyperparameters:")
    print("-"*30)
    print(f"| p_value_threshold  | {best_params['p_value_threshold']:.6f} |")
    print(f"| min_half_life     | {best_params['min_half_life']:>8d} |") 
    print(f"| volatility_window | {best_params['volatility_lookback']:>8d} |")
    print("-"*30 + "\n")
    
    # Step 2: Find cointegrated pairs using optimal parameters
    print("\nFinding cointegrated pairs...")
    pairs = find_cointegrated_pairs(
        tickers, 
        start_date, 
        end_date,
        best_params['p_value_threshold']
    )
    
    if not pairs:
        print("No cointegrated pairs found")
        return None
        
    print(f"Found {len(pairs)} cointegrated pairs")
    
    # Step 3: Optimize portfolio using Kelly criterion
    print("\nOptimizing portfolio allocations...")
    pairs_list = [(p[0], p[1], p[3]) for p in pairs]
    portfolio = optimize_kelly_pairs_portfolio(pairs_list, start_date, end_date)
    
    print("\nPortfolio allocations:")
    for pair, alloc in portfolio.items():
        print(f"\n{pair}:")
        print(f"Position size {alloc['ticker1']}: {alloc['position_size_1']:.3f}")
        print(f"Position size {alloc['ticker2']}: {alloc['position_size_2']:.3f}")
        print(f"Pair Sharpe ratio: {alloc['sharpe_ratio']:.2f}")
    
    return {
        'optimization_results': optimization_results,
        'cointegrated_pairs': pairs,
        'portfolio_allocations': portfolio
    }

if __name__ == "__main__":
    tickers = ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', 'MU', 'ASML']
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    results = run_pairs_trading_simulation(tickers, start_date, end_date)
