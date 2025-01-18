# scripts/train_rsi_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib , talib

def train_rsi_model(data_path, save_path):
    data = pd.read_csv(data_path)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)  # Calculate RSI

    # Prepare features (X) and target (y)
    data['Future_Close'] = data['Close'].shift(-1)  # Predict next day's Close
    data = data.dropna()
    X = data[['RSI']]
    y = data['Future_Close']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, save_path)
    print(f"RSI model saved to {save_path}")

# Run the training
if __name__ == "__main__":
    train_rsi_model(data_path='data/historical_data.csv', save_path='models/rsi_model.pkl')
