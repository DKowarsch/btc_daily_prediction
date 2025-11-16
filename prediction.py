# src/prediction.py
import pandas as pd
import numpy as np
import json  # ← ADD THIS IMPORT
import joblib  # ← ADD THIS IMPORT
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # ← ADD THIS IMPORT
from data_preparation import fetch_btc_daily_data, prepare_daily_timeseries_data

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

def main():
    """Main prediction function"""
    try:
        print("🚀 Starting BTC 7-Day Prediction")
        print("="*50)
        
        # Load model and metadata
        model_info, feature_columns, sequence_length, scaler, best_model = load_latest_model()
        
        # Fetch latest data
        print("📊 Fetching live BTC data...")
        raw_data = fetch_btc_daily_data()
        processed_data, _ = prepare_daily_timeseries_data(raw_data)
        
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

if __name__ == "__main__":
    main()