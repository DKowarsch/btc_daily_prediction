# src/report_generation.py - COMPLETE DASHBOARD VERSION
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

def create_model_performance_chart(performance_data):
    """Create model performance comparison chart"""
    models = list(performance_data.keys())
    metrics = ['test_accuracy', 'test_f1', 'test_profit_score', 'test_precision', 'test_recall']
    
    fig = go.Figure()
    
    # Professional color palette
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
    
    for i, metric in enumerate(metrics):
        values = [performance_data[model][metric] for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values,
            marker_color=colors[i],
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        height=400,
        template='plotly_white',
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        xaxis_title="Models",
        yaxis_title="Score",
        yaxis_tickformat=',.0%',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def generate_html_report():
    """Generate complete HTML dashboard with all sections"""
    print("Generating comprehensive dashboard...")
    
    # Load data
    performance_data, summary_data = load_performance_data()
    predictions_data = load_predictions()
    portfolio_data = load_portfolio_data()
    
    # Get best model info
    best_model = max(performance_data.items(), key=lambda x: x[1]['test_profit_score'])
    best_model_name = best_model[0]
    best_model_score = best_model[1]['test_profit_score']
    best_model_accuracy = best_model[1]['test_accuracy']
    
    # Create model performance chart
    model_chart = create_model_performance_chart(performance_data)
    
    # Generate HTML
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #2ecc71;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --dark-color: #2c3e50;
            --light-color: #f8f9fa;
            --gray-color: #7f8c8d;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
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
            padding: 40px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: var(--dark-color);
            font-size: 2.8em;
            margin-bottom: 10px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            color: var(--gray-color);
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        
        .timestamp {{
            display: inline-block;
            background: var(--light-color);
            padding: 12px 24px;
            border-radius: 50px;
            color: var(--gray-color);
            font-size: 0.9em;
            margin-top: 15px;
        }}
        
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        @media (max-width: 1200px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .card {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.15);
        }}
        
        .card h2 {{
            color: var(--dark-color);
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--light-color);
            font-weight: 600;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .card h2 i {{
            color: var(--primary-color);
        }}
        
        /* Portfolio Section */
        .portfolio-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .stat-box:hover {{
            transform: scale(1.05);
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: 700;
            margin: 10px 0 5px 0;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .stat-description {{
            font-size: 0.85em;
            opacity: 0.8;
            font-weight: 500;
        }}
        
        .section-subtitle {{
            color: var(--dark-color);
            font-size: 1.1em;
            font-weight: 600;
            margin: 25px 0 15px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .allocation-list {{
            list-style: none;
            margin: 20px 0;
        }}
        
        .allocation-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: var(--light-color);
            margin-bottom: 10px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }}
        
        .allocation-item:hover {{
            background: #e9ecef;
            transform: translateX(5px);
        }}
        
        .allocation-name {{
            font-weight: 500;
            color: var(--dark-color);
        }}
        
        .allocation-value {{
            font-weight: 600;
            color: var(--primary-color);
            font-size: 1.1em;
        }}
        
        .performance-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 20px;
        }}
        
        .performance-table th {{
            background: var(--light-color);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: var(--dark-color);
            border-bottom: 2px solid #dee2e6;
        }}
        
        .performance-table td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .improvement {{
            color: var(--success-color);
            font-weight: 600;
        }}
        
        .decline {{
            color: var(--danger-color);
            font-weight: 600;
        }}
        
        .download-btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-top: 15px;
        }}
        
        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }}
        
        /* Predictions Section */
        .current-price-container {{
            text-align: center;
            margin: 20px 0 30px 0;
        }}
        
        .current-price {{
            font-size: 3em;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 10px;
        }}
        
        .model-info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 25px 0;
        }}
        
        .model-stat {{
            background: var(--light-color);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .model-stat:hover {{
            background: #e9ecef;
            transform: translateY(-3px);
        }}
        
        .model-stat .value {{
            font-size: 1.4em;
            font-weight: 700;
            color: var(--dark-color);
            margin-top: 5px;
        }}
        
        .model-stat .label {{
            font-size: 0.9em;
            color: var(--gray-color);
            font-weight: 500;
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
            background: var(--light-color);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 3px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .prediction-day:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }}
        
        .prediction-day.up {{
            border-color: var(--success-color);
            background: linear-gradient(135deg, #ffffff 0%, #d5f4e6 100%);
        }}
        
        .prediction-day.down {{
            border-color: var(--danger-color);
            background: linear-gradient(135deg, #ffffff 0%, #fadbd8 100%);
        }}
        
        .day-number {{
            font-size: 1.3em;
            font-weight: 700;
            color: var(--dark-color);
            margin-bottom: 5px;
        }}
        
        .day-date {{
            font-size: 0.9em;
            color: var(--gray-color);
            margin-bottom: 15px;
        }}
        
        .day-price {{
            font-size: 1.5em;
            font-weight: 700;
            color: var(--dark-color);
            margin: 10px 0;
        }}
        
        .day-return {{
            font-size: 1.2em;
            font-weight: 600;
            margin: 10px 0;
        }}
        
        .return-positive {{
            color: var(--success-color);
        }}
        
        .return-negative {{
            color: var(--danger-color);
        }}
        
        .confidence-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .confidence-high {{
            background: var(--success-color);
            color: white;
        }}
        
        .confidence-medium {{
            background: var(--warning-color);
            color: white;
        }}
        
        .confidence-low {{
            background: var(--danger-color);
            color: white;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            padding: 30px;
            border-radius: 15px;
            margin-top: 30px;
            border-left: 5px solid var(--success-color);
        }}
        
        .recommendation-title {{
            color: #155724;
            font-size: 1.3em;
            font-weight: 700;
            margin-bottom: 15px;
        }}
        
        .recommendation-confidence {{
            display: inline-block;
            padding: 8px 16px;
            background: var(--success-color);
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .recommendation-text {{
            color: #155724;
            margin: 15px 0;
            line-height: 1.5;
        }}
        
        .return-summary {{
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 20px;
            color: #155724;
        }}
        
        /* Charts Section */
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-top: 30px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .chart-container h3 {{
            color: var(--dark-color);
            margin-bottom: 20px;
            font-size: 1.2em;
            font-weight: 600;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid rgba(255,255,255,0.1);
            color: white;
            font-size: 0.9em;
        }}
        
        .disclaimer {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 0.85em;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>Financial Analytics Dashboard</h1>
            <p class="subtitle">Riverside Data Solutions, LLC & DeepSeek-powered Portfolio Optimization & BTC Predictions</p>
            <div class="timestamp">
                <i class="fas fa-clock"></i> Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
        </div>
        
        <div class="main-grid">
            <!-- Left Column: Portfolio Optimization -->
            <div class="column">
                <div class="card">
                    <h2><i class="fas fa-chart-pie"></i> Portfolio Optimization</h2>
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
                    
                    <div class="section-subtitle">
                        <i class="fas fa-list-ol"></i> Optimal Portfolio Allocation
                    </div>
                    <ul class="allocation-list">
                    ''' + '\n'.join([f'''
                        <li class="allocation-item">
                            <span class="allocation-name">{asset}</span>
                            <span class="allocation-value">{weight*100:.1f}%</span>
                        </li>
                    ''' for asset, weight in portfolio_data['optimal_weights'].items() if weight > 0.01]) + '''
                    </ul>
                    
                    <div class="section-subtitle">
                        <i class="fas fa-chart-bar"></i> Performance Comparison
                    </div>
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
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="portfolio/optimization_results.json" class="download-btn" download>
                            <i class="fas fa-download"></i> Download Portfolio Data
                        </a>
                    </div>
                    ''' if portfolio_data else '<p style="text-align: center; color: var(--gray-color); padding: 40px;">No portfolio data available</p>'}
                </div>
            </div>
            
            <!-- Right Column: BTC Predictions -->
            <div class="column">
                <div class="card">
                    <h2><i class="fas fa-bitcoin"></i> BTC 7-Day Predictions</h2>
                    {f'''
                    <div class="current-price-container">
                        <div class="current-price">${predictions_data['current_price']:,.2f}</div>
                        <div style="color: var(--gray-color); font-size: 0.9em;">
                            <i class="fas fa-chart-line"></i> Real-time Bitcoin Price
                        </div>
                    </div>
                    
                    <div class="model-info-grid">
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
                            <div class="value" style="color: {'var(--success-color)' if predictions_data['trading_recommendation']['total_return'] >= 0 else 'var(--danger-color)'}">
                                {predictions_data['trading_recommendation']['total_return']:+.2f}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="section-subtitle">
                        <i class="fas fa-calendar-alt"></i> 7-Day Price Predictions
                    </div>
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
                            <i class="fas fa-chart-line"></i> 7-Day Return: <span style="color: {'var(--success-color)' if total_return >= 0 else 'var(--danger-color)'}">{total_return:+.2f}%</span> | 
                            <i class="fas fa-arrow-up"></i> Bullish: {bullish_days}/7 | 
                            <i class="fas fa-arrow-down"></i> Bearish: {bearish_days}/7
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="predictions/latest_predictions.json" class="download-btn" download style="margin-right: 10px;">
                            <i class="fas fa-download"></i> Download JSON
                        </a>
                        <a href="predictions/latest_predictions.csv" class="download-btn" download>
                            <i class="fas fa-file-csv"></i> Download CSV
                        </a>
                    </div>
                    '''.format(
                        recommendation=predictions_data['trading_recommendation']['recommendation'],
                        confidence=predictions_data['trading_recommendation']['confidence'],
                        reasoning=predictions_data['trading_recommendation']['reasoning'],
                        total_return=predictions_data['trading_recommendation']['total_return'],
                        bullish_days=predictions_data['trading_recommendation']['bullish_days'],
                        bearish_days=predictions_data['trading_recommendation']['bearish_days']
                    ) if predictions_data else '<p style="text-align: center; color: var(--gray-color); padding: 40px;">No prediction data available</p>'}
                </div>
            </div>
        </div>
        
        <!-- Charts Section -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3><i class="fas fa-chart-bar"></i> Model Performance Comparison</h3>
                <div id="model-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3><i class="fas fa-image"></i> Prediction Visualization</h3>
                <div style="text-align: center; margin: 20px 0;">
                    <img src="predictions/prediction_visualization.png" alt="BTC Prediction Chart" style="max-width: 100%; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <p style="margin-top: 15px; color: var(--gray-color); font-size: 0.9em;">
                        <i class="fas fa-info-circle"></i> Daily prediction chart showing price forecasts and confidence levels
                    </p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><i class="fas fa-robot"></i> Powered by Riverside Data Solutions, LLC & AI Models</p>
            <p><i class="fas fa-sync-alt"></i> Predictions updated daily at 15:00 UTC | Next update: <span id="next-update">Calculating...</span></p>
            <div class="disclaimer">
                <i class="fas fa-exclamation-triangle"></i> <strong>Investment Disclaimer:</strong> This dashboard provides financial analytics for informational purposes only. 
                It does not constitute investment advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
            </div>
        </div>
    </div>
    
    <script>
        // Countdown timer
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
        }});
    </script>
    
    <!-- Plotly Chart -->
    <script>
        {f'var modelChartData = {model_chart.to_json()};' if 'model_chart' in locals() else ''}
        
        {'''
        if (typeof modelChartData !== 'undefined') {
            Plotly.newPlot('model-chart', modelChartData.data, modelChartData.layout);
        }
        '''}
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
    
    print("✅ Complete dashboard generated: reports/dashboard.html and index.html")

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

## 📊 Portfolio Optimization
"""
    
    if portfolio_data:
        markdown_content += f"""
### Performance Metrics
- **Expected Return**: {portfolio_data['performance']['optimal_return']*100:.1f}%
- **Volatility**: {portfolio_data['performance']['optimal_volatility']*100:.1f}%
- **Sharpe Ratio**: {portfolio_data['performance']['optimal_sharpe']:.3f}

### Optimal Allocation
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
### Model Performance
- **Model**: {predictions_data['model_used']}
- **Profit Score**: {predictions_data['model_performance']['profit_score']:.4f}
- **Accuracy**: {predictions_data['model_performance']['accuracy']*100:.1f}%

### Trading Recommendation: {recommendation['recommendation']}
- **Confidence**: {recommendation['confidence']}
- **Reasoning**: {recommendation['reasoning']}
- **Total 7-Day Return**: {recommendation['total_return']:+.2f}%

### Daily Predictions
| Day | Date | Prediction | Return | Price | Confidence |
|-----|------|------------|--------|-------|------------|
"""
        
        for pred in predictions_data['predictions']:
            markdown_content += f"| {pred['day']} | {pred['date']} | {pred['predicted_direction']} | {pred['predicted_return']:+.2f}% | ${pred['predicted_price']:.2f} | {pred['confidence']} |\n"
    
    markdown_content += f"""

## 🤖 Model Performance Comparison
"""
    
    for model, metrics in performance_data.items():
        markdown_content += f"""
### {model}
- **Accuracy**: {metrics['test_accuracy']:.3f}
- **F1-Score**: {metrics['test_f1']:.3f}
- **Profit Score**: {metrics['test_profit_score']:.3f}
- **Precision**: {metrics['test_precision']:.3f}
- **Recall**: {metrics['test_recall']:.3f}
"""
    
    markdown_content += f"""

## 📁 Generated Files

### Portfolio
- `portfolio/optimization_results.json` - Portfolio optimization data

### Predictions
- `predictions/latest_predictions.json` - Latest predictions (JSON)
- `predictions/latest_predictions.csv` - Predictions in CSV format
- `predictions/prediction_visualization.png` - Prediction chart

### Models
- `models/best_model_info.json` - Best model configuration
- `reports/model_performance.json` - Detailed performance metrics
- `reports/dashboard.html` - Interactive dashboard

![Prediction Visualization](predictions/prediction_visualization.png)

---

*Automatically generated by Riverside Data Solutions AI System • Updated daily at 15:00 UTC*
"""
    
    # Save markdown report
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print("✅ Complete markdown report generated: README.md")

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
