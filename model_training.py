# model_training.py
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import datetime
import joblib

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
            valid_data[col] = valid_data[col].fillna(method='ffill').fillna(0)
    
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

def train_models():
    """Train models using the prepared data from data_preparation.py"""
    print("🚀 Starting BTC Day Trading Model Training")
    print("="*50)
    
    # Import the prepare function only (not fetch)
    from data_preparation import prepare_daily_timeseries_data
    
    # Load the raw data that was already fetched by data_preparation.py
    # We need to simulate what data_preparation.py already did
    print("📊 Loading prepared BTC data...")
    
    # Since data_preparation.py already ran and prepared the data,
    # we need to either:
    # 1. Load the processed data from a file, or
    # 2. Run the data preparation again but only the processing part
    
    # Option: Import and use the same data preparation
    from data_preparation import fetch_btc_daily_data
    raw_data = fetch_btc_daily_data()
    processed_data, feature_columns = prepare_daily_timeseries_data(raw_data)
    
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
    
    # Save the actual model and scaler
    training_results['best_model'].save('models/best_model.h5')
    joblib.dump(training_results['scaler'], 'models/scaler.pkl')
    
    print("✅ Training results saved!")

if __name__ == "__main__":
    try:
        # Train models with the prepared data
        training_results = train_models()
        
        # Save results for web dashboard
        save_training_results(training_results)
        
        print("\n🎉 Model training completed successfully!")
        print(f"📊 Best model: {training_results['best_model_name']}")
        print(f"💰 Test profit score: {training_results['training_summary']['best_profit_score']:.4f}")
        print(f"📈 Test accuracy: {training_results['training_summary']['best_accuracy']:.4f}")
        print(f"📁 Results saved to: models/ and reports/")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Create error report
        os.makedirs('reports', exist_ok=True)
        with open('reports/training_error.json', 'w') as f:
            json.dump({'error': str(e), 'timestamp': datetime.now().isoformat()}, f, indent=2)
        raise
