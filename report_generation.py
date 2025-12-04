# src/report_generation.py - TWO-COLUMN LAYOUT VERSION
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_performance_data():
    """Load model performance data"""
    try:
        with open('reports/model_performance.json', 'r') as f:
            performance = json.load(f)
        with open('reports/training_summary.json', 'r') as f:
            summary = json.load(f)
        return performance, summary
    except Exception as e:
        raise Exception(f"Failed to load performance data: {e}")

def load_predictions():
    """Load latest predictions"""
    try:
        with open('predictions/latest_predictions.json', 'r') as f:
            predictions = json.load(f)
        return predictions
    except Exception as e:
        print(f"Warning: Could not load predictions: {e}")
        return None

def load_portfolio_data():
    """Load portfolio optimization data"""
    try:
        with open('portfolio/optimization_results.json', 'r') as f:
            portfolio = json.load(f)
        return portfolio
    except Exception as e:
        print(f"Warning: Could not load portfolio data: {e}")
        return None

def generate_html_report():
    """Generate HTML report with two-column layout"""
    print("Generating professional dashboard...")
    
    # Load data
    performance_data, summary_data = load_performance_data()
    predictions_data = load_predictions()
    portfolio_data = load_portfolio_data()
    
    # Get best model info
    best_model = max(performance_data.items(), key=lambda x: x[1]['test_profit_score'])
    best_model_name = best_model[0]
    best_model_score = best_model[1]['test_profit_score']
    best_model_accuracy = best_model[1]['test_accuracy']
    
    # Generate HTML with two-column layout
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analytics Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        
        @media (max-width: 1200px) {{
            .content-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .column {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        
        .card {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: #2c3e50;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
            font-weight: 600;
            font-size: 1.3em;
        }}
        
        /* Portfolio Section Styles */
        .portfolio-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: 700;
            margin: 10px 0 5px 0;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .stat-description {{
            font-size: 0.8em;
            opacity: 0.8;
        }}
        
        .allocation-list {{
            list-style: none;
            margin-top: 20px;
        }}
        
        .allocation-item {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .allocation-item:last-child {{
            border-bottom: none;
        }}
        
        .allocation-name {{
            font-weight: 500;
            color: #2c3e50;
        }}
        
        .allocation-value {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .performance-table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        .performance-table td {{
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .improvement {{
            color: #2ecc71;
            font-weight: 600;
        }}
        
        .decline {{
            color: #e74c3c;
            font-weight: 600;
        }}
        
        /* Predictions Section Styles */
        .current-price {{
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin: 10px 0 30px 0;
        }}
        
        .model-info {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .model-stat {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .model-stat .value {{
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 5px;
        }}
        
        .model-stat .label {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        
        .predictions-grid {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 15px;
            margin: 30px 0;
        }}
        
        @media (max-width: 1400px) {{
            .predictions-grid {{
                grid-template-columns: repeat(4, 1fr);
            }}
        }}
        
        @media (max-width: 900px) {{
            .predictions-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        .prediction-day {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .prediction-day:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }}
        
        .prediction-day.up {{
            border-color: #2ecc71;
        }}
        
        .prediction-day.down {{
            border-color: #e74c3c;
        }}
        
        .day-number {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .day-date {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 15px;
        }}
        
        .day-price {{
            font-size: 1.3em;
            font-weight: 700;
            color: #2c3e50;
            margin: 10px 0;
        }}
        
        .day-return {{
            font-size: 1.1em;
            font-weight: 600;
            margin: 10px 0;
        }}
        
        .return-positive {{
            color: #2ecc71;
        }}
        
        .return-negative {{
            color: #e74c3c;
        }}
        
        .confidence-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .confidence-high {{
            background: #2ecc71;
            color: white;
        }}
        
        .confidence-medium {{
            background: #f39c12;
            color: white;
        }}
        
        .confidence-low {{
            background: #e74c3c;
            color: white;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            padding: 25px;
            border-radius: 12px;
            margin-top: 30px;
            border-left: 5px solid #28a745;
        }}
        
        .recommendation-title {{
            color: #155724;
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .recommendation-confidence {{
            display: inline-block;
            padding: 5px 10px;
            background: #28a745;
            color: white;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .recommendation-text {{
            color: #155724;
            margin: 10px 0;
        }}
        
        .return-summary {{
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 15px;
            color: #155724;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            color: white;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
        }}
        
        .badge-success {{
            background: #2ecc71;
            color: white;
        }}
        
        .badge-warning {{
            background: #f39c12;
            color: white;
        }}
        
        .badge-danger {{
            background: #e74c3c;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>Financial Analytics Dashboard</h1>
            <p class="subtitle">Riverside Data Solutions, LLC & DeepSeek-powered Portfolio Optimization & BTC Predictions</p>
            <div style="margin-top: 15px; font-size: 0.9em; color: #7f8c8d;">
                <i class="fas fa-clock"></i> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
        </div>
        
        <div class="content-grid">
            <!-- Left Column: Portfolio Optimization -->
            <div class="column">
                <div class="card">
                    <h2>Portfolio Optimization</h2>
                    {f'''
                    <div class="portfolio-stats">
                        <div class="stat-box">
                            <div class="stat-label">Optimal Volatility</div>
                            <div class="stat-value">{portfolio_data['performance']['optimal_volatility']*100:.1f}%</div>
                            <div class="stat-description">Lower Risk</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Expected Return</div>
                            <div class="stat-value">{portfolio_data['performance']['optimal_return']*100:.1f}%</div>
                            <div class="stat-description">Higher Return</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Sharpe Ratio</div>
                            <div class="stat-value">{portfolio_data['performance']['optimal_sharpe']:.3f}</div>
                            <div class="stat-description">Better Efficiency</div>
                        </div>
                    </div>
                    
                    <h3 style="margin-top: 30px; margin-bottom: 20px; color: #2c3e50; font-size: 1.1em; font-weight: 600;">Optimal Portfolio Allocation</h3>
                    <ul class="allocation-list">
                    ''' + '\n'.join([f'''
                        <li class="allocation-item">
                            <span class="allocation-name">{asset}</span>
                            <span class="allocation-value">{weight*100:.1f}%</span>
                        </li>
                    ''' for asset, weight in portfolio_data['optimal_weights'].items() if weight > 0.01]) + '''
                    </ul>
                    
                    <h3 style="margin-top: 30px; margin-bottom: 20px; color: #2c3e50; font-size: 1.1em; font-weight: 600;">Performance Comparison</h3>
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Optimal</th>
                                <th>Equal Weight</th>
                                <th>Improvement</th>
                            </tr>
                        </thead>
                        <tbody>
                    ''' + f'''
                            <tr>
                                <td>Volatility</td>
                                <td>{portfolio_data['performance']['optimal_volatility']*100:.1f}%</td>
                                <td>{portfolio_data['performance']['equal_volatility']*100:.1f}%</td>
                                <td class="improvement">+{(portfolio_data['performance']['equal_volatility'] - portfolio_data['performance']['optimal_volatility'])/portfolio_data['performance']['equal_volatility']*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Return</td>
                                <td>{portfolio_data['performance']['optimal_return']*100:.1f}%</td>
                                <td>{portfolio_data['performance']['equal_return']*100:.1f}%</td>
                                <td class="{ 'improvement' if portfolio_data['performance']['optimal_return'] >= portfolio_data['performance']['equal_return'] else 'decline' }">{((portfolio_data['performance']['optimal_return'] - portfolio_data['performance']['equal_return'])/portfolio_data['performance']['equal_return']*100):+.1f}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{portfolio_data['performance']['optimal_sharpe']:.3f}</td>
                                <td>{portfolio_data['performance']['equal_sharpe']:.3f}</td>
                                <td class="improvement">+{(portfolio_data['performance']['optimal_sharpe'] - portfolio_data['performance']['equal_sharpe'])/portfolio_data['performance']['equal_sharpe']*100:.1f}%</td>
                            </tr>
                    ''' + '''
                        </tbody>
                    </table>
                    
                    <div style="margin-top: 30px; text-align: center;">
                        <div style="display: inline-block; padding: 10px 20px; background: #667eea; color: white; border-radius: 25px; font-size: 0.9em; font-weight: 600;">
                            <i class="fas fa-chart-bar"></i> View Detailed Portfolio Data
                        </div>
                    </div>
                    ''' if portfolio_data else '<p style="color: #7f8c8d; text-align: center;">No portfolio data available</p>'}
                </div>
            </div>
            
            <!-- Right Column: BTC Predictions -->
            <div class="column">
                <div class="card">
                    <h2>BTC 7-Day Predictions</h2>
                    {f'''
                    <div class="current-price">${predictions_data['current_price']:,.2f}</div>
                    
                    <div class="model-info">
                        <div class="model-stat">
                            <div class="label">Model</div>
                            <div class="value">{predictions_data['model_used']}</div>
                        </div>
                        <div class="model-stat">
                            <div class="label">Profit Score</div>
                            <div class="value">{predictions_data['model_performance']['profit_score']:.4f}</div>
                        </div>
                        <div class="model-stat">
                            <div class="label">Accuracy</div>
                            <div class="value">{predictions_data['model_performance']['accuracy']*100:.1f}%</div>
                        </div>
                        <div class="model-stat">
                            <div class="label">Total Return</div>
                            <div class="value" style="color: {'#2ecc71' if predictions_data['trading_recommendation']['total_return'] >= 0 else '#e74c3c'}">
                                {predictions_data['trading_recommendation']['total_return']:+.2f}%
                            </div>
                        </div>
                    </div>
                    
                    <h3 style="margin-bottom: 20px; color: #2c3e50; font-size: 1.1em; font-weight: 600;">7-Day Price Predictions</h3>
                    <div class="predictions-grid">
                    ''' + '\n'.join([f'''
                        <div class="prediction-day {'up' if p['predicted_direction'] == 'UP' else 'down'}">
                            <div class="day-number">Day {p['day']}</div>
                            <div class="day-date">{p['date']}</div>
                            <div class="day-price">${p['predicted_price']:,.2f}</div>
                            <div class="day-return {'return-positive' if p['predicted_return'] >= 0 else 'return-negative'}">
                                {p['predicted_return']:+.2f}%
                            </div>
                            <div class="confidence-badge {'confidence-high' if p['confidence'] == 'HIGH' else 'confidence-medium' if p['confidence'] == 'MEDIUM' else 'confidence-low'}">
                                {p['confidence']}
                            </div>
                        </div>
                    ''' for p in predictions_data['predictions']]) + '''
                    </div>
                    
                    <div class="recommendation-box">
                        <div class="recommendation-title">{recommendation}</div>
                        <div class="recommendation-confidence">{confidence}</div>
                        <div class="recommendation-text">{reasoning}</div>
                        <div class="return-summary">
                            7-Day Return: <span style="color: {'#2ecc71' if total_return >= 0 else '#e74c3c'}">{total_return:+.2f}%</span> | 
                            Bullish: {bullish_days}/7 | Bearish: {bearish_days}/7
                        </div>
                    </div>
                    '''.format(
                        recommendation=predictions_data['trading_recommendation']['recommendation'],
                        confidence=predictions_data['trading_recommendation']['confidence'],
                        reasoning=predictions_data['trading_recommendation']['reasoning'],
                        total_return=predictions_data['trading_recommendation']['total_return'],
                        bullish_days=predictions_data['trading_recommendation']['bullish_days'],
                        bearish_days=predictions_data['trading_recommendation']['bearish_days']
                    ) if predictions_data else '<p style="color: #7f8c8d; text-align: center;">No prediction data available</p>'}
                </div>
                
                <!-- Visualization Section -->
                <div class="card">
                    <h2>Prediction Visualization</h2>
                    <div style="text-align: center; margin: 20px 0;">
                        <img src="predictions/prediction_visualization.png" alt="BTC Prediction Chart" style="max-width: 100%; border-radius: 10px; box-shadow: 0 5px 25px rgba(0,0,0,0.1);">
                        <p style="margin-top: 15px; color: #7f8c8d; font-size: 0.9em;">
                            <i class="fas fa-info-circle"></i> Daily prediction chart showing price forecasts and confidence levels
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><i class="fas fa-robot"></i> Powered by Riverside Data Solutions, LLC & AI Models</p>
            <p>Predictions updated daily at 15:00 UTC | Next update: <span id="next-update">Calculating...</span></p>
            <p style="margin-top: 15px; font-size: 0.8em; opacity: 0.7;">
                <i class="fas fa-exclamation-triangle"></i> Investment Disclaimer: This is for informational purposes only. Always do your own research.
            </p>
        </div>
    </div>
    
    <script>
        // Countdown timer for next update
        function calculateNextUpdate() {{
            const now = new Date();
            const nextUpdate = new Date();
            nextUpdate.setUTCHours(15, 0, 0, 0);
            if (now.getUTCHours() >= 15) {{
                nextUpdate.setUTCDate(nextUpdate.getUTCDate() + 1);
            }}
            const timeUntilUpdate = nextUpdate - now;
            const hours = Math.floor(timeUntilUpdate / (1000 * 60 * 60));
            const minutes = Math.floor((timeUntilUpdate % (1000 * 60 * 60)) / (1000 * 60));
            document.getElementById('next-update').textContent = `in ${{hours}}h ${{minutes}}m`;
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            calculateNextUpdate();
            setInterval(calculateNextUpdate, 60000);
            
            // Add smooth hover effects
            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {{
                card.style.transition = 'all 0.3s ease';
                card.addEventListener('mouseenter', () => {{
                    card.style.transform = 'translateY(-5px)';
                    card.style.boxShadow = '0 15px 40px rgba(0,0,0,0.15)';
                }});
                card.addEventListener('mouseleave', () => {{
                    card.style.transform = 'translateY(0)';
                    card.style.boxShadow = '0 10px 30px rgba(0,0,0,0.1)';
                }});
            }});
        }});
    </script>
</body>
</html>
'''
    
    # Save HTML report
    os.makedirs('reports', exist_ok=True)
    with open('reports/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Also create index.html for GitHub Pages
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Two-column dashboard generated: reports/dashboard.html and index.html")

def generate_markdown_report():
    """Generate markdown report for GitHub"""
    print("Generating markdown report...")
    
    performance_data, summary_data = load_performance_data()
    predictions_data = load_predictions()
    portfolio_data = load_portfolio_data()
    
    # Find best model
    best_model = max(performance_data.items(), key=lambda x: x[1]['test_profit_score'])
    
    markdown_content = f"""# Financial Analytics Dashboard

**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 📊 Portfolio Optimization Summary
"""
    
    if portfolio_data:
        markdown_content += f"""
- **Expected Return**: {portfolio_data['performance']['optimal_return']*100:.1f}%
- **Volatility**: {portfolio_data['performance']['optimal_volatility']*100:.1f}%
- **Sharpe Ratio**: {portfolio_data['performance']['optimal_sharpe']:.3f}

### Top Holdings:
"""
        for asset, weight in portfolio_data['optimal_weights'].items():
            if weight > 0.01:
                markdown_content += f"- **{asset}**: {weight*100:.1f}%\n"
    
    markdown_content += f"""

## 🚀 BTC 7-Day Predictions
"""
    
    if predictions_data:
        recommendation = predictions_data['trading_recommendation']
        
        markdown_content += f"""
### Current Price: ${predictions_data['current_price']:,.2f}
### Model: {predictions_data['model_used']}
- **Profit Score**: {predictions_data['model_performance']['profit_score']:.4f}
- **Accuracy**: {predictions_data['model_performance']['accuracy']*100:.1f}%

### Trading Recommendation: {recommendation['recommendation']}
- **Confidence**: {recommendation['confidence']}
- **Reasoning**: {recommendation['reasoning']}
- **Total 7-Day Return**: {recommendation['total_return']:+.2f}%

### Daily Predictions:
| Day | Date | Prediction | Return | Price | Confidence |
|-----|------|------------|--------|-------|------------|
"""
        
        for pred in predictions_data['predictions']:
            markdown_content += f"| {pred['day']} | {pred['date']} | {pred['predicted_direction']} | {pred['predicted_return']:+.2f}% | ${pred['predicted_price']:.2f} | {pred['confidence']} |\n"
    
    markdown_content += f"""

## 📁 Generated Files

- `portfolio/optimization_results.json` - Portfolio optimization data
- `models/best_model_info.json` - Best model configuration
- `reports/model_performance.json` - Detailed performance metrics
- `reports/dashboard.html` - Interactive dashboard
- `predictions/latest_predictions.json` - Latest predictions
- `predictions/latest_predictions.csv` - Predictions in CSV format
- `predictions/prediction_visualization.png` - Prediction chart

![Prediction Visualization](predictions/prediction_visualization.png)

---

*Automatically generated by Riverside Data Solutions AI System • Updated daily at 15:00 UTC*
"""
    
    # Save markdown report
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print("✅ Professional markdown report generated: README.md")

def main():
    """Generate all reports"""
    try:
        print("🚀 Generating Financial Analytics Dashboard")
        print("="*50)
        
        generate_html_report()
        generate_markdown_report()
        
        print("\n✅ All reports generated successfully!")
        print("📊 View interactive dashboard: reports/dashboard.html")
        print("📝 View summary report: README.md")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
