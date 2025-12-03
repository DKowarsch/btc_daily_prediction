# data_preparation.py
import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime, timedelta
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import scipy.optimize as optimize
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== PORTFOLIO OPTIMIZATION FUNCTIONS ====================

def fetch_portfolio_data():
    """Fetch data for stock and crypto portfolio"""
    stocks = ['AAPL', 'TSLA', 'AMD', 'IONQ', 'CRML']
    cryptos = ['dogecoin', 'bitcoin', 'ethereum', 'solana', 'binancecoin']
    
    crypto_display_names = {
        'dogecoin': 'DOGE', 'bitcoin': 'BTC', 'ethereum': 'ETH',
        'solana': 'SOL', 'binancecoin': 'BNB'
    }
    
    print("📊 Fetching portfolio data...")
    
    # Get stock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock_data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True, progress=False)
    stock_prices = stock_data['Close']
    
    # Get crypto data from CoinGecko
    def get_crypto_data(coin_id):
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': 365, 'interval': 'daily'}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                df.set_index('date', inplace=True)
                df = df[['price']]
                df.columns = [crypto_display_names[coin_id]]
                return df
        except:
            return None
    
    crypto_dfs = []
    for crypto in cryptos:
        crypto_df = get_crypto_data(crypto)
        if crypto_df is not None:
            crypto_dfs.append(crypto_df)
        time.sleep(1.2)
    
    crypto_prices = pd.concat(crypto_dfs, axis=1) if crypto_dfs else pd.DataFrame()
    
    # Align data
    if not crypto_prices.empty:
        stock_prices_date = stock_prices.copy()
        stock_prices_date.index = stock_prices_date.index.date
        crypto_prices_date = crypto_prices.copy()
        crypto_prices_date.index = pd.to_datetime(crypto_prices_date.index).date
        
        common_dates = stock_prices_date.index.intersection(crypto_prices_date.index)
        common_dates = sorted(common_dates)
        
        if len(common_dates) > 0:
            stock_aligned = stock_prices_date.loc[common_dates]
            crypto_aligned = crypto_prices_date.loc[common_dates]
            stock_aligned.index = pd.to_datetime(stock_aligned.index)
            crypto_aligned.index = pd.to_datetime(crypto_aligned.index)
            all_prices = pd.concat([stock_aligned, crypto_aligned], axis=1)
        else:
            all_prices = stock_prices
    else:
        all_prices = stock_prices
    
    all_prices = all_prices.dropna()
    returns = np.log(all_prices / all_prices.shift(1)).dropna()
    cov_matrix = returns.cov() * 252
    
    print(f"✅ Portfolio data: {all_prices.shape[1]} assets, {all_prices.shape[0]} days")
    return all_prices, returns, cov_matrix

def optimize_portfolio(cov_matrix, expected_returns):
    """Find minimum variance portfolio - FIXED"""
    # Remove the default parameter logic since we always pass expected_returns
    n_assets = len(cov_matrix)
    
    def portfolio_variance(weights):
        return weights @ cov_matrix.values @ weights
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1/n_assets] * n_assets)
    
    result = optimize.minimize(portfolio_variance, init_guess, 
                             method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        return init_guess

def run_portfolio_optimization():
    """Run portfolio optimization and display results"""
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMIZATION")
    print("="*50)
    
    all_prices, returns, cov_matrix = fetch_portfolio_data()
    expected_returns = returns.mean() * 252
    
    # Get optimal weights
    optimal_weights = optimize_portfolio(cov_matrix, expected_returns)
    
    # Calculate portfolio stats
    def portfolio_stats(weights):
        port_return = np.sum(weights * expected_returns)
        port_volatility = np.sqrt(weights @ cov_matrix.values @ weights)
        sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0
        return port_return, port_volatility, sharpe_ratio
    
    opt_return, opt_vol, opt_sharpe = portfolio_stats(optimal_weights)
    
    # Equal weight comparison
    equal_weights = np.array([1/len(cov_matrix)] * len(cov_matrix))
    eq_return, eq_vol, eq_sharpe = portfolio_stats(equal_weights)
    
    # FIX: Add epsilon to avoid division by zero
    epsilon = 1e-10
    return_improvement = (opt_return - eq_return) / (eq_return + epsilon) * 100
    vol_improvement = (eq_vol - opt_vol) / (eq_vol + epsilon) * 100
    sharpe_improvement = (opt_sharpe - eq_sharpe) / (eq_sharpe + epsilon) * 100
    
    print("\n🏆 OPTIMAL PORTFOLIO ALLOCATION:")
    print("-" * 40)
    for i, asset in enumerate(cov_matrix.index):
        if optimal_weights[i] > 0.01:  # Only show weights > 1%
            print(f"  {asset}: {optimal_weights[i]:.3f} ({optimal_weights[i]*100:.1f}%)")
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print("-" * 40)
    print(f"{'Metric':<15} {'Optimal':<12} {'Equal':<10} {'Improvement'}")
    print("-" * 40)
    print(f"{'Return':<15} {opt_return:.4f}{'':<8} {eq_return:.4f}{'':<8} {return_improvement:+.1f}%")
    print(f"{'Volatility':<15} {opt_vol:.4f}{'':<8} {eq_vol:.4f}{'':<8} {vol_improvement:+.1f}%")
    print(f"{'Sharpe Ratio':<15} {opt_sharpe:.4f}{'':<8} {eq_sharpe:.4f}{'':<8} {sharpe_improvement:+.1f}%")
    
    # Save portfolio results
    portfolio_results = {
        'optimal_weights': dict(zip(cov_matrix.index.tolist(), optimal_weights.tolist())),
        'performance': {
            'optimal_volatility': float(opt_vol),
            'optimal_return': float(opt_return),
            'optimal_sharpe': float(opt_sharpe),
            'equal_volatility': float(eq_vol),
            'equal_return': float(eq_return),
            'equal_sharpe': float(eq_sharpe)
        },
        'generated_at': datetime.now().isoformat()
    }
    
    os.makedirs('portfolio', exist_ok=True)
    with open('portfolio/optimization_results.json', 'w') as f:
        json.dump(portfolio_results, f, indent=2)
    
    print(f"\n✅ Portfolio results saved: portfolio/optimization_results.json")
    return portfolio_results

# ==================== BTC PREDICTION FUNCTIONS ====================

def fetch_btc_daily_data():
    """Fetch 5 years of daily BTC data"""
    print("📊 Fetching 5 years of BTC daily data...")
    try:
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="5y", interval="1d")  
        print(f"✅ Data fetched: {len(hist)} samples from {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
        return hist
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def prepare_daily_timeseries_data(data):
    """Prepare technical indicators and features for the model"""
    if data.empty:
        print("❌ No data to process!")
        return pd.DataFrame(), []
        
    processed_data = data.copy()

    print("🔧 Calculating technical indicators...")
    print(f"Total samples: {len(processed_data)}")

    # Ensure index is proper DatetimeIndex
    if not isinstance(processed_data.index, pd.DatetimeIndex):
        print("⚠️ Converting index to DatetimeIndex...")
        processed_data.index = pd.to_datetime(processed_data.index)

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
    print(f"📈 Samples with 1-day targets: {samples_with_targets}")
    
    if samples_with_targets > 0:
        target_data = processed_data[processed_data['Target_1d_Up'].notna()]
        target_dist = target_data['Target_1d_Up'].value_counts(normalize=True) * 100
        print(f"🎯 Target distribution: UP {target_dist.get(1, 0):.1f}%, DOWN {target_dist.get(0, 0):.1f}%")

    return processed_data, available_features

def calculate_profit_score(y_true, y_pred, y_pred_proba):
    """Calculate a profit-oriented score"""
    accuracy = accuracy_score(y_true, y_pred)
    
    correct_predictions = (y_true == y_pred)
    if np.sum(correct_predictions) > 0:
        confidence_score = np.mean(y_pred_proba[correct_predictions])
    else:
        confidence_score = 0
    
    false_positive_rate = np.sum((y_pred == 1) & (y_true == 0)) / len(y_true)
    false_negative_rate = np.sum((y_pred == 0) & (y_true == 1)) / len(y_true)
    
    profit_score = (
        accuracy * 0.3 +
        confidence_score * 0.4 +
        (1 - false_positive_rate) * 0.2 +
        (1 - false_negative_rate) * 0.1
    )
    
    return profit_score

def build_gru_tiny_model(input_shape):
    """Simple GRU model"""
    model = Sequential([
        GRU(8, input_shape=input_shape, return_sequences=False, dropout=0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01),
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_lstm_complex_model(input_shape):
    """Complex LSTM model"""
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=True, dropout=0.2),
        LSTM(16, return_sequences=False, dropout=0.2),
        Dense(8, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_bilstm_model(input_shape):
    """Bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(16, return_sequences=False, dropout=0.2), input_shape=input_shape),
        Dense(8, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_dense_wide_model(input_shape):
    """Wide dense model"""
    flattened_size = input_shape[0] * input_shape[1]
    model = Sequential([
        Dense(128, activation='relu', input_shape=(flattened_size,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.002),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def create_robust_sequences(data, feature_columns, target_col='Target_1d_Up', sequence_length=30):
    """Create sequences with proper time series validation"""
    valid_data = data[data[target_col].notna()].copy()
    
    for col in feature_columns:
        if valid_data[col].isna().sum() > 0:
            valid_data[col] = valid_data[col].ffill().fillna(0)
    
    X, y = [], []
    for i in range(sequence_length, len(valid_data)):
        sequence_data = valid_data[feature_columns].iloc[i-sequence_length:i]
        if not sequence_data.isna().any().any():
            X.append(sequence_data.values)
            y.append(valid_data[target_col].iloc[i])
    
    return np.array(X), np.array(y)

def scale_features_properly(X_train, X_val, X_test):
    """Scale features without data leakage"""
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler.transform(X_val_flat)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_models_live():
    """Train models using live data"""
    print("🚀 Starting BTC Day Trading Model Training")
    print("="*50)
    
    raw_data = fetch_btc_daily_data()
    processed_data, feature_columns = prepare_daily_timeseries_data(raw_data)
    
    if processed_data.empty:
        raise Exception("❌ No data available for training!")
    
    print(f"✅ Data loaded: {len(processed_data)} samples, {len(feature_columns)} features")
    
    sequence_length = 10
    X, y = create_robust_sequences(processed_data, feature_columns, 'Target_1d_Up', sequence_length)
    
    n = len(X)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"📈 Data split: Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features_properly(X_train, X_val, X_test)
    
    input_shape = (sequence_length, len(feature_columns))
    models = {
        'GRU_Tiny': build_gru_tiny_model(input_shape),
        'LSTM_Complex': build_lstm_complex_model(input_shape),
        'BiLSTM': build_bilstm_model(input_shape),
        'Dense_Wide': build_dense_wide_model(input_shape)
    }
    
    performance = {}
    trained_models = {}
    
    print("\n🏃 Training models...")
    
    for name, model in models.items():
        print(f"🎯 Training {name}...")
        
        try:
            if 'Dense' in name:
                X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
                X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
                X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
                
                history = model.fit(X_train_flat, y_train, validation_data=(X_val_flat, y_val),
                                  epochs=50, batch_size=32, verbose=0,
                                  callbacks=[EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy')])
                
                train_pred_proba = model.predict(X_train_flat, verbose=0).flatten()
                val_pred_proba = model.predict(X_val_flat, verbose=0).flatten()
                test_pred_proba = model.predict(X_test_flat, verbose=0).flatten()
                
            else:
                history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                                  epochs=50, batch_size=32, verbose=0,
                                  callbacks=[EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy')])
                
                train_pred_proba = model.predict(X_train_scaled, verbose=0).flatten()
                val_pred_proba = model.predict(X_val_scaled, verbose=0).flatten()
                test_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            
            train_pred = (train_pred_proba > 0.5).astype(int)
            val_pred = (val_pred_proba > 0.5).astype(int)
            test_pred = (test_pred_proba > 0.5).astype(int)
            
            test_acc = accuracy_score(y_test, test_pred)
            test_precision = precision_score(y_test, test_pred, zero_division=0)
            test_recall = recall_score(y_test, test_pred, zero_division=0)
            test_f1 = f1_score(y_test, test_pred, zero_division=0)
            test_profit_score = calculate_profit_score(y_test, test_pred, test_pred_proba)
            
            performance[name] = {
                'test_accuracy': float(test_acc),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
                'test_f1': float(test_f1),
                'test_profit_score': float(test_profit_score),
                'training_date': datetime.now().isoformat(),
                'parameters': model.count_params()
            }
            
            trained_models[name] = model
            print(f"   ✅ Test Profit Score: {test_profit_score:.4f}, Accuracy: {test_acc:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if performance:
        best_model_name = max(performance.items(), key=lambda x: x[1]['test_profit_score'])[0]
        best_profit_score = performance[best_model_name]['test_profit_score']
        best_accuracy = performance[best_model_name]['test_accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"💰 Test Profit Score: {best_profit_score:.4f}")
        print(f"📈 Test Accuracy: {best_accuracy:.4f}")
        
        return {
            'best_model': trained_models[best_model_name],
            'best_model_name': best_model_name,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'sequence_length': sequence_length,
            'performance': performance,
            'trained_models': trained_models,
            'latest_data': processed_data,
            'training_summary': {
                'total_samples': len(processed_data),
                'training_date': datetime.now().isoformat(),
                'best_profit_score': best_profit_score,
                'best_accuracy': best_accuracy,
                'models_trained': len(trained_models)
            }
        }
    else:
        raise Exception("❌ No models were successfully trained")

def save_training_results(training_results):
    """Save training results"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    with open('reports/model_performance.json', 'w') as f:
        json.dump(training_results['performance'], f, indent=2)
    
    with open('reports/training_summary.json', 'w') as f:
        json.dump(training_results['training_summary'], f, indent=2)
    
    with open('models/best_model_info.json', 'w') as f:
        json.dump({
            'best_model_name': training_results['best_model_name'],
            'feature_columns': training_results['feature_columns'],
            'sequence_length': training_results['sequence_length'],
            'training_date': training_results['training_summary']['training_date'],
            'best_profit_score': training_results['training_summary']['best_profit_score'],
            'best_accuracy': training_results['training_summary']['best_accuracy']
        }, f, indent=2)
    
    training_results['best_model'].save('models/best_model.h5')
    joblib.dump(training_results['scaler'], 'models/scaler.pkl')
    
    print("✅ Training results saved!")

def load_latest_model():
    """Load the latest trained model"""
    try:
        with open('models/best_model_info.json', 'r') as f:
            model_info = json.load(f)
        
        feature_columns = model_info['feature_columns']
        sequence_length = model_info['sequence_length']
        scaler = joblib.load('models/scaler.pkl')
        best_model = load_model('models/best_model.h5', compile=False)
        best_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        
        print(f"✅ Loaded model: {model_info['best_model_name']}")
        print(f"📊 Features: {len(feature_columns)}, Sequence: {sequence_length}")
        
        return model_info, feature_columns, sequence_length, scaler, best_model
        
    except Exception as e:
        raise Exception(f"❌ Failed to load model: {e}")

def predict_7_days_ahead(model, scaler, feature_columns, sequence_length, processed_data, model_info):
    """Generate 7-day predictions using recursive forecasting"""
    try:
        print("🔮 Generating 7-day predictions...")
        
        predictions = []
        current_sequence = processed_data[feature_columns].tail(sequence_length).copy()
        
        # Get current price for reference
        current_price = processed_data['Close'].iloc[-1]
        simulated_price = current_price
        
        for day in range(1, 8):
            # Prepare current sequence - handle missing values
            current_sequence_filled = current_sequence.copy()
            for col in feature_columns:
                if current_sequence_filled[col].isna().any():
                    current_sequence_filled[col] = current_sequence_filled[col].ffill().fillna(0)
            
            # Scale the sequence
# ✅ CORRECT: This keeps the shape as (10, 19) for proper scaling
            sequence_reshaped = current_sequence_filled.values  # Shape: (10, 19)
            sequence_scaled = scaler.transform(sequence_reshaped)  # Shape: (10, 19)
                        
            # Reshape based on model type
            if 'Dense' in model_info['best_model_name']:
                prediction_input = sequence_scaled.reshape(1, -1)
            else:
                prediction_input = sequence_scaled.reshape(1, sequence_length, len(feature_columns))
            
            # Make prediction
            prediction_proba = model.predict(prediction_input, verbose=0)
            
            # Handle different output formats
            if hasattr(prediction_proba, '__len__') and len(prediction_proba) > 0:
                prediction_proba = float(prediction_proba[0][0] if len(prediction_proba[0]) > 0 else prediction_proba[0])
            else:
                prediction_proba = float(prediction_proba)
                
            prediction = 1 if prediction_proba > 0.5 else 0
            
            # Calculate predicted price movement
            recent_volatility = processed_data['Returns'].tail(10).std()
            base_move = recent_volatility * 0.8  # 80% of recent volatility
            
            if prediction == 1:
                predicted_return = base_move * prediction_proba
            else:
                predicted_return = -base_move * (1 - prediction_proba)
            
            simulated_price = simulated_price * (1 + predicted_return / 100)
            
            predictions.append({
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_direction': 'UP' if prediction == 1 else 'DOWN',
                'predicted_probability': float(prediction_proba),
                'predicted_price': float(simulated_price),
                'predicted_return': float(predicted_return),
                'confidence': 'HIGH' if prediction_proba > 0.7 or prediction_proba < 0.3 else 'MEDIUM' if prediction_proba > 0.6 or prediction_proba < 0.4 else 'LOW'
            })
            
            # Update sequence for next prediction
            if day < 7:
                new_row = current_sequence.iloc[-1:].copy()
                price_change_pct = predicted_return
                new_row['Returns'] = price_change_pct
                new_row['Log_Returns'] = np.log(1 + price_change_pct / 100) * 100
                new_row['Momentum_1period'] = price_change_pct / 100
                
                # Update the sequence
                current_sequence = pd.concat([current_sequence.iloc[1:], new_row], axis=0)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error in 7-day prediction: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_prediction():
    """Run 7-day prediction using trained model"""
    try:
        print("🚀 Starting BTC 7-Day Prediction")
        print("="*50)
        
        model_info, feature_columns, sequence_length, scaler, best_model = load_latest_model()
        
        print("📊 Fetching live BTC data...")
        raw_data = fetch_btc_daily_data()
        processed_data, _ = prepare_daily_timeseries_data(raw_data)
        
        if processed_data.empty:
            raise Exception("❌ No data available for prediction!")
            
        print(f"✅ Data loaded: {len(processed_data)} samples")
        
        # Generate 7-day predictions
        predictions = predict_7_days_ahead(best_model, scaler, feature_columns, sequence_length, processed_data, model_info)
        
        if not predictions:
            raise Exception("❌ Failed to generate predictions")
        
        current_price = processed_data['Close'].iloc[-1]
        
        # Calculate overall recommendation
        up_predictions = sum(1 for p in predictions if p['predicted_direction'] == 'UP')
        total_return = sum(p['predicted_return'] for p in predictions)
        
        if up_predictions >= 5:
            overall_recommendation = "STRONG BUY"
            reasoning = f"Bullish trend with {up_predictions}/7 days predicted UP"
        elif up_predictions >= 3:
            overall_recommendation = "HOLD"
            reasoning = f"Mixed signals with {up_predictions}/7 days predicted UP"
        else:
            overall_recommendation = "SELL"
            reasoning = f"Bearish trend with only {up_predictions}/7 days predicted UP"
        
        # Create prediction results
        prediction_data = {
            'current_price': float(current_price),
            'predictions': predictions,
            'trading_recommendation': {
                'recommendation': overall_recommendation,
                'confidence': 'HIGH' if abs(total_return) > 10 else 'MEDIUM',
                'reasoning': reasoning,
                'total_return': float(total_return),
                'bullish_days': up_predictions,
                'bearish_days': 7 - up_predictions
            },
            'model_used': model_info['best_model_name'],
            'model_performance': {
                'profit_score': model_info['best_profit_score'],
                'accuracy': model_info['best_accuracy']
            },
            'generated_at': datetime.now().isoformat()
        }
        
        # Save predictions
        os.makedirs('predictions', exist_ok=True)
        with open('predictions/latest_predictions.json', 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        # Also save as CSV
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('predictions/latest_predictions.csv', index=False)
        
        # Display results
        print(f"\n🏆 Using best model: {model_info['best_model_name']}")
        print(f"💰 Model profit score: {model_info['best_profit_score']:.4f}")
        print(f"📈 Model accuracy: {model_info['best_accuracy']:.4f}")
        print(f"📊 Current BTC Price: ${current_price:,.2f}")
        print(f"\n🎯 7-DAY PREDICTIONS:")
        print("-" * 50)
        for pred in predictions:
            direction_icon = "🟢" if pred['predicted_direction'] == 'UP' else "🔴"
            print(f"Day {pred['day']} ({pred['date']}): {direction_icon} {pred['predicted_direction']}")
            print(f"   Price: ${pred['predicted_price']:,.2f} | Return: {pred['predicted_return']:+.2f}%")
            print(f"   Confidence: {pred['confidence']} | Probability: {pred['predicted_probability']:.1%}")
            print()
        
        print(f"📊 SUMMARY: {overall_recommendation}")
        print(f"💡 Reasoning: {reasoning}")
        print(f"📈 Total 7-day return: {total_return:+.2f}%")
        
        return prediction_data
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        print(f"🔍 Detailed error: {traceback.format_exc()}")
        return False
    

# ==================== MAIN FUNCTION ====================

def main():
    """Main function - runs portfolio optimization and BTC prediction"""
    try:
        print("🚀 STARTING COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        # Run portfolio optimization
        portfolio_results = run_portfolio_optimization()
        
        print("\n" + "="*60)
        
        # Run BTC prediction
        training_results = train_models_live()
        save_training_results(training_results)
        
        print(f"\n🎉 Model training completed!")
        print(f"📊 Best model: {training_results['best_model_name']}")
        print(f"💰 Test profit score: {training_results['training_summary']['best_profit_score']:.4f}")
        
        run_prediction()
        
        print(f"\n📁 All results saved:")
        print(f"   - Portfolio: portfolio/optimization_results.json")
        print(f"   - Models: models/")
        print(f"   - Reports: reports/")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
