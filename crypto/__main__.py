import base64
import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

app = Flask(__name__)

BYBIT_API_URL = "https://api.bybit.com/v5/market/kline"

def get_bybit_data(symbol, interval, start_time, end_time):
    start_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_time = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": start_time,
        "end": end_time
    }
    response = requests.get(BYBIT_API_URL, params=params)
    data = response.json()
    if data["retCode"] == 0:
        df = pd.DataFrame(data["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["close"] = pd.to_numeric(df["close"])
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        return df
    else:
        return None

def calculate_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_lstm_prediction(df):
    df["close_scaled"] = MinMaxScaler().fit_transform(df[["close"]])
    X, y = [], []
    for i in range(50, len(df)):
        X.append(df["close_scaled"].iloc[i-50:i].values)
        y.append(df["close_scaled"].iloc[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    prediction = model.predict(X[-1].reshape(1, X.shape[1], 1))
    return "UP" if prediction > df["close_scaled"].iloc[-1] else "DOWN"

def analyze_data(df):
    df["rsi"] = calculate_rsi(df["close"], 14)
    trend_signal = generate_lstm_prediction(df)

    signals = []
    if df["sma_20"].iloc[-1] > df["sma_50"].iloc[-1]:
        signals.append("LONG Signal (SMA)")
    elif df["sma_20"].iloc[-1] < df["sma_50"].iloc[-1]:
        signals.append("SHORT Signal (SMA)")

    if df["rsi"].iloc[-1] < 30:
        signals.append("Oversold - Consider LONG (RSI)")
    elif df["rsi"].iloc[-1] > 70:
        signals.append("Overbought - Consider SHORT (RSI)")

    signals.append(f"Trend Prediction (LSTM): {trend_signal}")
    return signals

def plot_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    
    plt.figure(figsize=(10,5))
    plt.plot(df["timestamp"], df["close"], label="Close Price", color="blue")
    plt.plot(df["timestamp"], df["sma_20"], label="SMA 20", color="green")
    plt.plot(df["timestamp"], df["sma_50"], label="SMA 50", color="red")
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Bybit Market Data with Analysis")
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    signals = None
    if request.method == 'POST':
        symbol = request.form['symbol']
        interval = request.form['interval']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        
        df = get_bybit_data(symbol, interval, start_time, end_time)
        if df is not None:
            plot_url = plot_data(df)
            signals = analyze_data(df)
    
    return render_template('index.html', plot_url=plot_url, signals=signals)

if __name__ == '__main__':
    app.run(debug=True)
