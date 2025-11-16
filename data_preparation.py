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
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA PREPARATION FUNCTIONS ====================

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

# ==================== MODEL TRAINING FUNCTIONS ====================

def calculate_profit_score(y_true, y_pred, y_pred_proba):
    """
    Calculate a profit-oriented score that rewards:
    - High-confidence correct predictions (winning trades)
    - Penalizes false positives (losing trades) more heavily
    - Rewards recall (capturing opportunities)
    """
    # Base accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confidence-weighted accuracy (higher confidence correct predictions)
    correct_predictions = (y_true == y_pred)
    if np.sum(correct_predictions) > 0:
        confidence_score = np.mean(y_pred_proba[correct_predictions])
    else:
        confidence_score = 0
    
    # Asymmetric penalty: false positives cost money, false misses cost opportunity
    false_positive_rate = np.sum((y_pred == 1) & (y_true == 0)) / len(y_true)
    false_negative_rate = np.sum((y_pred == 0) & (y_true == 1)) / len(y_true)
    
    # Trading-focused scoring (adjust weights based on your strategy)
    profit_score = (
        accuracy * 0.3 +                    # Base correctness
        confidence_score * 0.4 +            # Confidence in wins
        (1 - false_positive_rate) * 0.2 +   # Avoid losing trades
        (1 - false_negative_rate) * 0.1     # Capture opportunities
    )
    
    return profit_score

def build_gru_tiny_model(input_shape):
    """Simple GRU model - often works best for financial data"""
    model = Sequential([
        GRU(8, input_shape=input_shape, return_sequences=False, dropout=0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01),
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_lstm_complex_model(input_shape):
    """Complex LSTM model for comparison"""
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
    """Wide dense model as baseline"""
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
    # Use only rows with valid targets
    valid_data = data[data[target_col].notna()].copy()
    
    # Handle any remaining NaN values
    for col in feature_columns:
        if valid_data[col].isna().sum() > 0:
            valid_data[col] = valid_data[col].ffill().fillna(0)
    
    X, y = [], []
    for i in range(sequence_length, len(valid_data)):
        # Only include sequences without NaN values
        sequence_data = valid_data[feature_columns].iloc[i-sequence_length:i]
        if not sequence_data.isna().any().any():
            X.append(sequence_data.values)
            y.append(valid_data[target_col].iloc[i])
    
    return np.array(X), np.array(y)

def scale_features_properly(X_train, X_val, X_test):
    """Scale features without data leakage"""
    # Reshape for scaling
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    
    # Transform validation and test data using training scaler
    X_val_scaled_flat = scaler.transform(X_val_flat)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    
    # Reshape back to sequences
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_models_live():
    """Train models using live data and return comprehensive results"""
    print("🚀 Starting BTC Day Trading Model Training")
    print("="*50)
    
    # Fetch and prepare fresh data
    print("📊 Fetching live BTC data...")
    raw_data = fetch_btc_daily_data()
    processed_data, feature_columns = prepare_daily_timeseries_data(raw_data)
    
    if processed_data.empty:
        raise Exception("❌ No data available for training!")
    
    print(f"✅ Data loaded: {len(processed_data)} samples, {len(feature_columns)} features")
    
    # Create sequences
    sequence_length = 10
    X, y = create_robust_sequences(processed_data, feature_columns, 'Target_1d_Up', sequence_length)
    
    # Chronological split (60-20-20)
    n = len(X)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"📈 Data split:")
    print(f"   Train: {X_train.shape[0]} sequences")
    print(f"   Val:   {X_val.shape[0]} sequences")
    print(f"   Test:  {X_test.shape[0]} sequences")
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features_properly(X_train, X_val, X_test)
    
    # Build diverse models
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
                
                history = model.fit(
                    X_train_flat, y_train,
                    validation_data=(X_val_flat, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy')
                    ]
                )
                
                # Get predictions
                train_pred_proba = model.predict(X_train_flat, verbose=0).flatten()
                val_pred_proba = model.predict(X_val_flat, verbose=0).flatten()
                test_pred_proba = model.predict(X_test_flat, verbose=0).flatten()
                
            else:
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy')
                    ]
                )
                
                # Get predictions
                train_pred_proba = model.predict(X_train_scaled, verbose=0).flatten()
                val_pred_proba = model.predict(X_val_scaled, verbose=0).flatten()
                test_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            
            # Convert to binary predictions
            train_pred = (train_pred_proba > 0.5).astype(int)
            val_pred = (val_pred_proba > 0.5).astype(int)
            test_pred = (test_pred_proba > 0.5).astype(int)
            
            # Calculate comprehensive metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            test_precision = precision_score(y_test, test_pred, zero_division=0)
            test_recall = recall_score(y_test, test_pred, zero_division=0)
            test_f1 = f1_score(y_test, test_pred, zero_division=0)
            
            # Calculate profit score
            test_profit_score = calculate_profit_score(y_test, test_pred, test_pred_proba)
            
            # Prediction statistics
            pred_mean = np.mean(test_pred_proba)
            pred_std = np.std(test_pred_proba)
            confidence = np.mean(np.abs(test_pred_proba - 0.5)) * 2
            
            performance[name] = {
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'test_accuracy': float(test_acc),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
                'test_f1': float(test_f1),
                'test_profit_score': float(test_profit_score),
                'prediction_mean': float(pred_mean),
                'prediction_std': float(pred_std),
                'confidence': float(confidence),
                'training_date': datetime.now().isoformat(),
                'parameters': model.count_params()
            }
            
            trained_models[name] = model
            print(f"   ✅ Test Profit Score: {test_profit_score:.4f}, Accuracy: {test_acc:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Determine best model by PROFIT SCORE
    if performance:
        best_model_name = max(performance.items(), key=lambda x: x[1]['test_profit_score'])[0]
        best_profit_score = performance[best_model_name]['test_profit_score']
        best_accuracy = performance[best_model_name]['test_accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"💰 Test Profit Score: {best_profit_score:.4f}")
        print(f"📈 Test Accuracy: {best_accuracy:.4f}")
        
        if best_profit_score > 0.6:
            print("✅ Excellent profit potential")
        elif best_profit_score > 0.55:
            print("⚠️  Good profit potential")
        elif best_profit_score > 0.5:
            print("📊 Moderate profit potential")
        else:
            print("❌ Limited profit potential")
        
        # Return comprehensive results
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
    """Save training results for web dashboard"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Save performance report
    with open('reports/model_performance.json', 'w') as f:
        json.dump(training_results['performance'], f, indent=2)
    
    # Save training summary
    with open('reports/training_summary.json', 'w') as f:
        json.dump(training_results['training_summary'], f, indent=2)
    
    # Save best model info
    with open('models/best_model_info.json', 'w') as f:
        json.dump({
            'best_model_name': training_results['best_model_name'],
            'feature_columns': training_results['feature_columns'],
            'sequence_length': training_results['sequence_length'],
            'training_date': training_results['training_summary']['training_date'],
            'best_profit_score': training_results['training_summary']['best_profit_score'],
            'best_accuracy': training_results['training_summary']['best_accuracy']
        }, f, indent=2)
    
    # Save the best model
    best_model = training_results['best_model']
    best_model.save('models/best_model.h5')
    
    # Save scaler for predictions
    joblib.dump(training_results['scaler'], 'models/scaler.pkl')
    
    print("✅ Training results saved!")

# ==================== PREDICTION FUNCTIONS ====================

def load_latest_model():
    """Load the latest trained model and associated files"""
    try:
        with open('models/best_model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load feature columns and sequence length
        feature_columns = model_info['feature_columns']
        sequence_length = model_info['sequence_length']
        
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Load the actual model
        best_model = load_model('models/best_model.h5')
        
        print(f"✅ Loaded model: {model_info['best_model_name']}")
        print(f"📊 Features: {len(feature_columns)}, Sequence: {sequence_length}")
        
        return model_info, feature_columns, sequence_length, scaler, best_model
        
    except Exception as e:
        raise Exception(f"❌ Failed to load model: {e}")

def predict_next_7_days(model, model_name, processed_data, feature_columns, scaler, sequence_length=12, days_to_predict=7):
    """
    Predict next 7 days using recursive forecasting starting from last available data
    """
    print(f"\n{'='*80}")
    print("7-DAY FUTURE PREDICTION")
    print(f"{'='*80}")

    # Get the most recent data
    last_data_date = processed_data.index[-1]
    last_known_price = processed_data['Close'].iloc[-1]
    
    print(f"PREDICTION STARTING POINT:")
    print(f"  Last available date: {last_data_date}")
    print(f"  Last known price: ${last_known_price:.2f}")
    print(f"  Predicting next {days_to_predict} days")
    print(f"  Using {len(feature_columns)} features")
    print(f"  Model: {model_name}")

    # Prepare the most recent sequence for prediction
    recent_data = processed_data[feature_columns].iloc[-sequence_length:].copy()
    
    # Handle any missing values
    if recent_data.isna().any().any():
        print("⚠️  Warning: Missing values in recent data, filling with zeros")
        recent_data = recent_data.fillna(0)
    
    # Scale the recent sequence
    scaled_sequence = scaler.transform(recent_data)
    
    print(f"  Using sequence from {processed_data.index[-sequence_length]} to {last_data_date}")

    # Generate future dates
    future_dates = []
    for i in range(1, days_to_predict + 1):
        future_date = last_data_date + timedelta(days=i)
        future_dates.append(future_date)

    # Recursive prediction
    current_sequence = scaled_sequence.copy()
    future_predictions = []
    future_prices = [last_known_price]
    current_price = last_known_price

    print(f"\nMAKING PREDICTIONS:")
    print(f"{'Day':<4} {'Date':<12} {'Pred Return%':<14} {'Pred Price':<12}")
    print(f"{'-'*50}")

    for day in range(days_to_predict):
        # Handle different model types
        if 'Dense' in model_name:
            # Flatten input for dense model
            model_input = current_sequence.reshape(1, -1)  # Flatten to 2D
        else:
            # Keep 3D shape for LSTM/GRU models
            model_input = current_sequence.reshape(1, sequence_length, len(feature_columns))
        
        # Predict next day's return
        try:
            predicted_return = model.predict(model_input, verbose=0)[0, 0]
            future_predictions.append(predicted_return)
            
            # Calculate next price
            next_price = current_price * (1 + predicted_return / 100)
            future_prices.append(next_price)
            
            print(f"{day+1:<4} {future_dates[day].strftime('%Y-%m-%d'):<12} {predicted_return:+.4f}%{'':<8} ${next_price:.2f}")

            # Update sequence for next prediction
            # For simplicity, we just roll the window with the new prediction
            new_row = current_sequence[-1].copy()
            # Update the return feature (assuming it's the first feature)
            if len(feature_columns) > 0:
                new_row[0] = predicted_return
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            current_price = next_price
            
        except Exception as e:
            print(f"❌ Prediction error on day {day+1}: {e}")
            # Use a simple fallback
            fallback_return = 0.0  # Assume no change
            future_predictions.append(fallback_return)
            next_price = current_price
            future_prices.append(next_price)
            print(f"{day+1:<4} {future_dates[day].strftime('%Y-%m-%d'):<12} {fallback_return:+.4f}%{'':<8} ${next_price:.2f} (fallback)")

    return future_predictions, future_prices[1:], future_dates, last_known_price, last_data_date

def generate_trading_recommendation(future_returns, future_prices, last_known_price):
    """
    Generate trading recommendations based on 7-day predictions
    """
    print(f"\n{'='*80}")
    print("TRADING RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Calculate key metrics
    total_return = (future_prices[-1] - last_known_price) / last_known_price * 100
    avg_daily_return = np.mean(future_returns)
    bullish_days = sum(1 for ret in future_returns if ret > 0)
    bearish_days = len(future_returns) - bullish_days
    
    print(f"PREDICTION SUMMARY:")
    print(f"  Starting Price: ${last_known_price:.2f}")
    print(f"  Predicted 7-day Return: {total_return:+.2f}%")
    print(f"  Predicted End Price: ${future_prices[-1]:.2f}")
    print(f"  Bullish Days: {bullish_days}/7")
    print(f"  Bearish Days: {bearish_days}/7")
    print(f"  Average Daily Return: {avg_daily_return:+.4f}%")
    
    # Generate recommendation
    if total_return > 5:
        recommendation = "🟢 STRONG BUY"
        reasoning = "Strong bullish momentum expected"
        confidence = "High"
    elif total_return > 2:
        recommendation = "🟢 BUY"
        reasoning = "Positive momentum expected"
        confidence = "Medium-High"
    elif total_return > -2:
        recommendation = "⚪ NEUTRAL"
        reasoning = "Mixed signals, sideways movement likely"
        confidence = "Medium"
    elif total_return > -5:
        recommendation = "🟡 CAUTION"
        reasoning = "Moderate bearish pressure"
        confidence = "Medium"
    else:
        recommendation = "🔴 BEARISH"
        reasoning = "Strong downward momentum"
        confidence = "High"
    
    print(f"\nRECOMMENDATION: {recommendation}")
    print(f"Confidence: {confidence}")
    print(f"Reasoning: {reasoning}")
    
    # Risk management
    print(f"\nRISK MANAGEMENT:")
    stop_loss = last_known_price * 0.95
    take_profit = last_known_price * 1.08
    print(f"  Stop Loss: ${stop_loss:.0f} (5% down)")
    print(f"  Take Profit: ${take_profit:.0f} (8% up)")
    print(f"  Position Size: 1-2% of portfolio")
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'reasoning': reasoning,
        'total_return': total_return,
        'bullish_days': bullish_days,
        'bearish_days': bearish_days,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

def create_prediction_visualization(future_returns, future_prices, future_dates, last_price, last_date, processed_data, model_name):
    """Create comprehensive prediction visualization"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Recent prices + predictions
    plt.subplot(2, 2, 1)
    
    # Show last 30 days for context
    recent_days = min(30, len(processed_data))
    recent_dates = processed_data.index[-recent_days:]
    recent_prices = processed_data['Close'].iloc[-recent_days:]
    
    plt.plot(recent_dates, recent_prices, 'b-', label='Historical Prices', linewidth=2)
    plt.plot(future_dates, future_prices, 'r-', marker='o', label='Predicted Prices', linewidth=2)
    plt.axvline(last_date, color='green', linestyle='--', label='Prediction Start')
    
    plt.title(f'7-Day Price Prediction\nCurrent: ${last_price:.2f} → Predicted: ${future_prices[-1]:.2f}', 
              fontweight='bold')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 2: Predicted daily returns
    plt.subplot(2, 2, 2)
    
    colors = ['green' if r > 0 else 'red' for r in future_returns]
    bars = plt.bar(range(1, 8), future_returns, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, ret) in enumerate(zip(bars, future_returns)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if ret > 0 else -0.02), 
                f'{ret:+.2f}%', ha='center', va='bottom' if ret > 0 else 'top', fontweight='bold')
    
    plt.title('Predicted Daily Returns (%)', fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative return
    plt.subplot(2, 2, 3)
    
    cumulative_returns = [0]
    current_cumulative = 0
    for ret in future_returns:
        current_cumulative += ret
        cumulative_returns.append(current_cumulative)
    
    days_display = [last_date] + future_dates
    plt.plot(days_display, cumulative_returns, color='purple', marker='s', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Cumulative Return Forecast', fontweight='bold')
    plt.ylabel('Cumulative Return (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 4: Price targets
    plt.subplot(2, 2, 4)
    
    price_levels = [last_price] + future_prices
    plt.plot([last_date] + future_dates, price_levels, color='orange', marker='o', linewidth=2)
    
    # Add price labels
    for i, (date, price) in enumerate(zip([last_date] + future_dates, price_levels)):
        if i == 0:
            plt.annotate(f'Start\n${price:.0f}', (date, price), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        elif i == len(price_levels) - 1:
            plt.annotate(f'Target\n${price:.0f}', (date, price), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold', color='red')
    
    plt.title('Price Target Progression', fontweight='bold')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.suptitle(f'BTC 7-Day Prediction - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('predictions', exist_ok=True)
    plt.savefig('predictions/prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: predictions/prediction_visualization.png")

def save_prediction_results(future_returns, future_prices, future_dates, last_price, last_date, recommendation, model_name):
    """Save prediction results for web dashboard"""
    os.makedirs('predictions', exist_ok=True)
    
    prediction_data = {
        'generated_at': datetime.now().isoformat(),
        'model_used': model_name,
        'prediction_start': last_date.strftime('%Y-%m-%d'),
        'last_known_price': float(last_price),
        'predictions': [],
        'trading_recommendation': recommendation,
        'summary': {
            'total_7day_return': float((future_prices[-1] - last_price) / last_price * 100),
            'bullish_days': int(recommendation['bullish_days']),
            'bearish_days': int(recommendation['bearish_days']),
            'average_daily_return': float(np.mean(future_returns))
        }
    }
    
    for i, (date, ret, price) in enumerate(zip(future_dates, future_returns, future_prices)):
        prediction_data['predictions'].append({
            'day': i + 1,
            'date': date.strftime('%Y-%m-%d'),
            'predicted_return': float(ret),
            'predicted_price': float(price),
            'direction': 'UP' if ret > 0 else 'DOWN'
        })
    
    # Save as JSON
    with open('predictions/latest_predictions.json', 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    # Save as CSV for easy viewing
    predictions_df = pd.DataFrame(prediction_data['predictions'])
    predictions_df.to_csv('predictions/latest_predictions.csv', index=False)
    
    print("✅ Prediction results saved to predictions/latest_predictions.json")

def run_prediction():
    """Run prediction using trained model"""
    try:
        print("🚀 Starting BTC 7-Day Prediction")
        print("="*50)
        
        # Load model and metadata
        model_info, feature_columns, sequence_length, scaler, best_model = load_latest_model()
        
        # Fetch latest data
        print("📊 Fetching live BTC data...")
        raw_data = fetch_btc_daily_data()
        processed_data, _ = prepare_daily_timeseries_data(raw_data)
        
        if processed_data.empty:
            raise Exception("❌ No data available for prediction!")
            
        print(f"✅ Data loaded: {len(processed_data)} samples")
        
        # Generate 7-day predictions with actual model
        future_returns, future_prices, future_dates, last_price, last_date = predict_next_7_days(
            model=best_model,
            model_name=model_info['best_model_name'],
            processed_data=processed_data,
            feature_columns=feature_columns,
            scaler=scaler,
            sequence_length=sequence_length,
            days_to_predict=7
        )
        
        # Generate trading recommendations
        recommendation = generate_trading_recommendation(future_returns, future_prices, last_price)

        # Create visualization
        create_prediction_visualization(
            future_returns, future_prices, future_dates, last_price, last_date,
            processed_data, model_info['best_model_name']
        )

        # Save results
        save_prediction_results(
            future_returns, future_prices, future_dates, last_price, last_date,
            recommendation, model_info['best_model_name']
        )

        # Print methodology
        print(f"\n{'='*80}")
        print("PREDICTION METHODOLOGY")
        print(f"{'='*80}")
        print(f"✓ Best Model: {model_info['best_model_name']} (selected by profit score)")
        print(f"✓ Sequence length: {sequence_length} days")
        print(f"✓ Features: {len(feature_columns)} technical indicators")
        print(f"✓ Starting from: {last_date}")
        print(f"✓ Prediction period: 7 days")
        print("✓ Recursive forecasting with proper model input handling")
        print(f"✓ Model selection: Profit-focused (score: {model_info['best_profit_score']:.3f})")

        print(f"\n🎉 7-Day prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise

# ==================== MAIN EXECUTION ====================

def main():
    """Main function - runs training and prediction"""
    try:
        # Train models
        training_results = train_models_live()
        save_training_results(training_results)
        
        print("\n🎉 Model training completed successfully!")
        print(f"📊 Best model: {training_results['best_model_name']}")
        print(f"💰 Test profit score: {training_results['training_summary']['best_profit_score']:.4f}")
        print(f"📈 Test accuracy: {training_results['training_summary']['best_accuracy']:.4f}")
        
        # Run prediction
        run_prediction()
        
        print(f"\n📁 All results saved to: models/, reports/, predictions/")
        
    except Exception as e:
        print(f"❌ Combined process failed: {e}")
        raise

if __name__ == "__main__":
    main()
