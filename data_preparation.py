# src/data_preparation.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def fetch_btc_daily_data():
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(period="5y", interval="1d")  
    return hist

def load_and_prepare_daily_data(fetched_data, start_date=None, end_date=None):

    if fetched_data is None or fetched_data.empty:
        raise ValueError("BTC daily data is None or empty")

    print("✓ Using direct daily BTC data from Yahoo Finance")

    # Convert index to datetime if it's not already
    if not isinstance(fetched_data.index, pd.DatetimeIndex):
        fetched_data.index = pd.to_datetime(fetched_data.index)
    # Set default dates if not provided
    if start_date is None:
        start_date = fetched_data.index.min()
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()

    # Ensure ALL dates are in the same timezone (convert to UTC to match the data)
    start_date = pd.Timestamp(start_date).tz_localize('UTC')
    end_date = pd.Timestamp(end_date).tz_localize('UTC')
    data_start = fetched_data.index.min()
    data_end = fetched_data.index.max()

    # Use minimum of the two dates for start and end
    start_date = min(start_date, data_start)
    end_date = min(end_date, pd.Timestamp.today().tz_localize('UTC'))
    end_date = min(end_date, data_end)  # Also don't exceed available data

    print(f"Filtering from: {start_date} to {end_date}")

    # Filter the data
    mask = (fetched_data.index >= start_date) & (fetched_data.index <= end_date)
    filtered_data = fetched_data.loc[mask].copy()
    return filtered_data

# Load data without date filtering (use all available data)

candle = fetch_btc_daily_data()
print("Data loaded successfully!")

# Show the actual date range we're working with
print(f"\nACTUAL DATASET INFORMATION:")
print(f"  Start date: {candle.index[0]}")
print(f"  End date: {candle.index[-1]}")
print(f"  Total periods: {len(candle)} daily periods")
print(f"  Date range: {(candle.index[-1] - candle.index[0]).days} days")
print(f"  Data frequency: daily")
print(f"  Recent prices:")
for i in range(min(5, len(candle))):
    idx = -(i+1)
    print(f"    {candle.index[idx].strftime('%Y-%m-%d')}: ${candle['Close'].iloc[idx]:.2f}")


def prepare_daily_timeseries_data(data):
    processed_data = data.copy()

    print("Calculating technical indicators for DAY TRADING...")
    print(f"Total samples: {len(processed_data)}")

    # Calculate returns and technical indicators
    processed_data['Returns'] = processed_data['Close'].pct_change() * 100
    processed_data['Log_Returns'] = np.log(processed_data['Close'] / processed_data['Close'].shift(1)) * 100

    # SHORTER-TERM volatility for day trading
    processed_data['Volatility_3period'] = processed_data['Returns'].rolling(3, min_periods=1).std().ffill()  
    processed_data['Volatility_7period'] = processed_data['Returns'].rolling(7, min_periods=1).std().ffill()    
    processed_data['Volatility_14period'] = processed_data['Returns'].rolling(14, min_periods=1).std().ffill()  

    # SHORTER-TERM moving averages for day trading
    period_windows = [3, 7, 14]  
    for window in period_windows:
        processed_data[f'MA_{window}'] = processed_data['Close'].rolling(window, min_periods=1).mean().ffill()
        processed_data[f'MA_Ratio_{window}'] = (processed_data['Close'] / processed_data[f'MA_{window}'] - 1).ffill()

    # Volume indicators
    if 'Volume' in processed_data.columns:
        processed_data['Volume_MA_5'] = processed_data['Volume'].rolling(5, min_periods=1).mean().ffill()
        processed_data['Volume_Ratio'] = (processed_data['Volume'] / processed_data['Volume_MA_5']).ffill()
        processed_data['Volume_Spike'] = (processed_data['Volume_Ratio'] > 2).astype(int)
    else:
        processed_data['Volume_Ratio'] = 1.0
        processed_data['Volume_Spike'] = 0

    # SHORTER-TERM price momentum for day trading
    processed_data['Momentum_1period'] = (processed_data['Close'] / processed_data['Close'].shift(1) - 1).ffill()   
    processed_data['Momentum_3period'] = (processed_data['Close'] / processed_data['Close'].shift(3) - 1).ffill()    
    processed_data['Momentum_7period'] = (processed_data['Close'] / processed_data['Close'].shift(7) - 1).ffill()    

    # Time-based features
    processed_data['DayOfWeek'] = processed_data.index.dayofweek
    processed_data['Day_Sin'] = np.sin(2 * np.pi * processed_data['DayOfWeek'] / 7)
    processed_data['Day_Cos'] = np.cos(2 * np.pi * processed_data['DayOfWeek'] / 7)

    # RSI - standard 14 days
    def compute_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    processed_data['RSI_14'] = compute_rsi(processed_data['Close'], 14).ffill()

    # Price range features
    if 'High' in processed_data.columns and 'Low' in processed_data.columns:
        processed_data['Price_Range'] = (processed_data['High'] - processed_data['Low']) / processed_data['Close'] * 100
        processed_data['Range_MA_5'] = processed_data['Price_Range'].rolling(5, min_periods=1).mean().ffill()
    else:
        processed_data['Price_Range'] = 0.0
        processed_data['Range_MA_5'] = 0.0

    # SHORTER-TERM price levels for day trading
    processed_data['Price_7d_High'] = processed_data['Close'].rolling(7, min_periods=1).max().ffill()   # 1 week high
    processed_data['Price_7d_Low'] = processed_data['Close'].rolling(7, min_periods=1).min().ffill()    # 1 week low
    processed_data['Price_7d_Ratio'] = (processed_data['Close'] - processed_data['Price_7d_Low']) / (processed_data['Price_7d_High'] - processed_data['Price_7d_Low']).replace(0, 1e-8)

    # 1-day future return direction
    future_periods = 1  # NEXT DAY prediction for day trading
    processed_data['Future_1d_Return'] = (processed_data['Close'].shift(-future_periods) - processed_data['Close']) / processed_data['Close'] * 100
    processed_data['Target_1d_Up'] = (processed_data['Future_1d_Return'] > 0).astype(int)

    # Select features for day trading
    feature_columns = [
        'Returns', 'Log_Returns', 
        'Volatility_3period', 'Volatility_7period', 'Volatility_14period',
        'MA_Ratio_3', 'MA_Ratio_7', 'MA_Ratio_14',
        'Volume_Ratio', 'Volume_Spike', 
        'Momentum_1period', 'Momentum_3period', 'Momentum_7period', 
        'RSI_14', 'Price_Range', 'Range_MA_5', 'Price_7d_Ratio',
        'Day_Sin', 'Day_Cos'
    ]

    # Verify all features exist
    available_features = []
    for col in feature_columns:
        if col in processed_data.columns and not processed_data[col].isna().all():
            available_features.append(col)
        else:
            print(f"Warning: Feature '{col}' not available, skipping")

    print(f"DAY TRADING setup: {len(processed_data)} samples, {len(available_features)} features")
    print(f"Target: 1-day price direction (Target_1d_Up)")
    
    # Show target distribution
    samples_with_targets = processed_data['Target_1d_Up'].notna().sum()
    print(f"Samples with 1-day targets: {samples_with_targets}")
    
    if samples_with_targets > 0:
        target_data = processed_data[processed_data['Target_1d_Up'].notna()]
        target_dist = target_data['Target_1d_Up'].value_counts(normalize=True) * 100
        print(f"Target distribution: UP {target_dist.get(1, 0):.1f}%, DOWN {target_dist.get(0, 0):.1f}%")

    return processed_data, available_features

# Main execution when file is run directly
if __name__ == "__main__":
    print("🚀 Starting BTC Data Preparation")
    print("=" * 50)
    
    # Fetch and prepare data
    raw_data = fetch_btc_daily_data()
    processed_data, feature_columns = prepare_daily_timeseries_data(raw_data)
    
