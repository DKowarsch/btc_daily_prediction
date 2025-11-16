# src/data_preparation.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def fetch_btc_daily_data():
    """Fetch BTC daily data from Yahoo Finance"""
    print("📊 Fetching BTC daily data...")
    try:
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="5y", interval="1d")  
        print(f"✅ Successfully fetched {len(hist)} days of BTC data")
        return hist
    except Exception as e:
        print(f"❌ Error fetching BTC data: {e}")
        # Return sample data if fetch fails
        return create_sample_data()

def create_sample_data():
    """Create sample data if Yahoo Finance fails"""
    print("🔄 Creating sample data for testing...")
    dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
    np.random.seed(42)
    
    # Generate realistic BTC price data
    prices = [5000]
    for i in range(1, len(dates)):
        change = np.random.normal(0.002, 0.04)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))
    
    sample_data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.05)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.03)) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1e9, 5e10) for _ in prices],
        'Adj Close': prices
    }, index=dates)
    
    print(f"✅ Created sample data with {len(sample_data)} days")
    return sample_data

def prepare_daily_timeseries_data(data):
    """Prepare features and targets for time series prediction"""
    if data is None or data.empty:
        raise ValueError("Data is None or empty")
    
    processed_data = data.copy()
    
    # Ensure we have a datetime index
    if not isinstance(processed_data.index, pd.DatetimeIndex):
        processed_data.index = pd.to_datetime(processed_data.index)
    
    print("🔄 Calculating technical indicators...")
    print(f"📈 Total samples: {len(processed_data)}")

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
    processed_data['Price_7d_High'] = processed_data['Close'].rolling(7, min_periods=1).max().ffill()
    processed_data['Price_7d_Low'] = processed_data['Close'].rolling(7, min_periods=1).min().ffill()
    processed_data['Price_7d_Ratio'] = (processed_data['Close'] - processed_data['Price_7d_Low']) / (processed_data['Price_7d_High'] - processed_data['Price_7d_Low']).replace(0, 1e-8)

    # 1-day future return direction
    future_periods = 1
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
            print(f"⚠️ Warning: Feature '{col}' not available, skipping")

    print(f"✅ DAY TRADING setup: {len(processed_data)} samples, {len(available_features)} features")
    
    # Show target distribution
    samples_with_targets = processed_data['Target_1d_Up'].notna().sum()
    print(f"📊 Samples with 1-day targets: {samples_with_targets}")
    
    if samples_with_targets > 0:
        target_data = processed_data[processed_data['Target_1d_Up'].notna()]
        target_dist = target_data['Target_1d_Up'].value_counts(normalize=True) * 100
        print(f"🎯 Target distribution: UP {target_dist.get(1, 0):.1f}%, DOWN {target_dist.get(0, 0):.1f}%")

    return processed_data, available_features

# Main execution when file is run directly
if __name__ == "__main__":
    print("🚀 Starting BTC Data Preparation")
    print("=" * 50)
    
    # Fetch and prepare data
    raw_data = fetch_btc_daily_data()
    processed_data, feature_columns = prepare_daily_timeseries_data(raw_data)
    
    print(f"\n✅ Data preparation completed!")
    print(f"📈 Final dataset: {len(processed_data)} samples")
    print(f"🔧 Features: {len(feature_columns)}")
    print(f"🎯 Target: Target_1d_Up")
