# scripts/combine_predictions.py
import pandas as pd
import joblib, talib


def combine_predictions(data_path, rsi_model_path, candlestick_model_path):
    data = pd.read_csv(data_path)

    # Load models
    rsi_model = joblib.load(rsi_model_path)
    candlestick_model = joblib.load(candlestick_model_path)

    # Calculate RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

    # Prepare features
    rsi_features = data[['RSI']].dropna()
    candlestick_features = data[['High_Low', 'Open_Close']].dropna()

    # Ensure matching indices
    predictions = pd.DataFrame(index=rsi_features.index)
    predictions['RSI_Pred'] = rsi_model.predict(rsi_features)
    predictions['Candle_Pred'] = candlestick_model.predict(candlestick_features)

    # Combine predictions (e.g., weighted average)
    predictions['Final_Pred'] = (predictions['RSI_Pred'] + predictions['Candle_Pred']) / 2
    print(predictions)

if __name__ == "__main__":
    combine_predictions(
        data_path='data/historical_data.csv',
        rsi_model_path='models/rsi_model.pkl',
        candlestick_model_path='models/candlestick_model.pkl'
    )
