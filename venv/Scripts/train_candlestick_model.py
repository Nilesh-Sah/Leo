import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import joblib


# scripts/train_candlestick_model.py
def train_candlestick_model(data_path, save_path):
    data = pd.read_csv(data_path)

    # Prepare candlestick features (e.g., Open/Close ratios, High-Low differences)
    data['High_Low'] = data['High'] - data['Low']
    data['Open_Close'] = data['Close'] - data['Open']
    data['Future_Close'] = data['Close'].shift(-1)
    data = data.dropna()

    X = data[['High_Low', 'Open_Close']]
    y = data['Future_Close']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, save_path)
    print(f"Candlestick model saved to {save_path}")

if __name__ == "__main__":
    train_candlestick_model(data_path='data/historical_data.csv', save_path='models/candlestick_model.pkl')
