# Crypto Market Data Visualization and Analysis

This Flask application fetches market data from the **Bybit API**, processes it with **SMA, RSI, and LSTM predictions**, and visualizes the results in an interactive plot.

## Features
- Fetches historical market data from Bybit
- Computes **Simple Moving Averages (SMA)** and **Relative Strength Index (RSI)**
- Uses **LSTM Neural Networks** for trend prediction
- Displays an interactive plot with market indicators
- Provides trading signals based on analysis

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/KubiakJakub01/crypto-analysis.git
cd crypto-analysis
```

### 2. Install dependencies
First, create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python -m crypto
```

The application will be available at **http://127.0.0.1:5000/**.

## Usage
1. Enter the **symbol** (e.g., `BTCUSDT`, `ETHUSDT`).
2. Select the **interval** (e.g., `1`, `5`, `15`, `60` minutes).
3. Input **start and end time** in the format `YYYY-MM-DD HH:MM:SS`.
4. Click **Fetch & Analyze** to retrieve market data.
5. View the plot with indicators and trading signals.

## Example Input
```
Symbol: BTCUSDT
Interval: 15
Start Time: 2024-01-01 00:00:00
End Time: 2024-01-02 00:00:00
```

## Dependencies
- Flask
- Requests
- Matplotlib
- Pandas
- NumPy
- scikit-learn
- TensorFlow

These are listed in **requirements.txt**.
