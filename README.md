# pairs-trader

A Python-based pairs trading system that uses statistical arbitrage techniques to identify and trade cointegrated asset pairs. The system optimizes trading parameters using Bayesian optimization and sizes positions using the Kelly criterion.

## Features
- Cointegration analysis to find statistically linked asset pairs
- Bayesian optimization of strategy parameters using Optuna
- Kelly criterion position sizing for optimal capital allocation
- Historical backtesting and performance analysis

## Running the code

Install uv and run ```uv sync``` to install the dependencies. Then run the code with:

```
python -m src/simulate.py
```
