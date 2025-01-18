# scripts/rsi_strategy.py

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load historical data from CSV."""
    data = pd.read_csv(file_path)
    return data

def calculate_rsi(data, period=14):
    """Calculate RSI using TA-Lib."""
    close_prices = data['Close'].values
    rsi = talib.RSI(close_prices, timeperiod=period)
    return rsi

def plot_rsi(data, rsi):
    """Plot RSI and visualize overbought/oversold levels."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], rsi, label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (>70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (<30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()

def main():
    # Load data
    file_path = 'data/historical_data.csv'  # Adjust file path as necessary
    data = load_data(file_path)

    # Calculate RSI
    rsi_period = 14  # RSI calculation period
    rsi = calculate_rsi(data, period=rsi_period)

    # Plot RSI
    plot_rsi(data, rsi)

if __name__ == "__main__":
    main()
