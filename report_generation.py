# src/report_generation.py
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime
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

def create_performance_dashboard(performance_data):
    """Create model performance comparison dashboard"""
    models = list(performance_data.keys())
    metrics = ['test_accuracy', 'test_f1', 'test_profit_score', 'test_precision', 'test_recall']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Accuracy', 'F1-Score', 'Profit Score', 'Precision', 'Recall', 'Confidence'],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add bars for each metric
    for i, metric in enumerate(metrics):
        values = [performance_data[model][metric] for model in models]
        row = i // 3 + 1
        col = i % 3 + 1
        
        fig.add_trace(
            go.Bar(name=metric, x=models, y=values, 
                   text=[f'{v:.3f}' for v in values], textposition='auto'),
            row=row, col=col
        )
    
    # Add confidence metric
    confidence_values = [performance_data[model]['confidence'] for model in models]
    fig.add_trace(
        go.Bar(name='Confidence', x=models, y=confidence_values,
               text=[f'{v:.3f}' for v in confidence_values], textposition='auto'),
        row=2, col=3
    )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_prediction_timeline(predictions_data):
    """Create prediction timeline visualization"""
    if not predictions_data:
        return None
        
    predictions_df = pd.DataFrame(predictions_data['predictions'])
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    fig = go.Figure()
    
    # Add confidence bars
    for _, row in predictions_df.iterrows():
        color = 'green' if row['direction'] == 'UP' else 'red'
        fig.add_trace(go.Bar(
            x=[row['date']],
            y=[row['predicted_return']],
            name=row['direction'],
            marker_color=color,
            text=f"{row['direction']} ({row['predicted_return']:+.2f}%)",
            textposition='auto',
            hovertemplate=(
                f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                f"Prediction: {row['direction']}<br>"
                f"Return: {row['predicted_return']:+.2f}%<br>"
                f"Price: ${row['predicted_price']:.2f}"
            )
        ))
    
    fig.update_layout(
        title='BTC Price Predictions - Next 7 Days',
        xaxis_title='Date',
        yaxis_title='Predicted Return (%)',
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_training_history_chart(performance_data):
    """Create training history and parameter comparison"""
    models = list(performance_data.keys())
    parameters = [performance_data[model]['parameters'] for model in models]
    accuracy = [performance_data[model]['test_accuracy'] for model in models]
    profit_scores = [performance_data[model]['test_profit_score'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Size vs Accuracy', 'Profit Score Distribution'],
        specs=[[{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Model size vs accuracy
    for i, model in enumerate(models):
        fig.add_trace(
            go.Scatter(
                x=[parameters[i]],
                y=[accuracy[i]],
                mode='markers+text',
                marker=dict(size=20),
                text=[model],
                textposition="middle right",
                name=model
            ),
            row=1, col=1
        )
    
    # Profit score distribution
    fig.add_trace(
        go.Pie(
            labels=models,
            values=profit_scores,
            hole=0.4,
            hoverinfo='label+value+percent'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Model Characteristics Analysis',
        height=400,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Number of Parameters', row=1, col=1)
    fig.update_yaxes(title_text='Test Accuracy', row=1, col=1)
    
    return fig

# In report_generation.py, update the generate_html_report() function:

def generate_html_report():
    """Generate complete HTML report"""
    print("Generating performance report...")
    
    # Load data
    performance_data, summary_data = load_performance_data()
    predictions_data = load_predictions()
    
    # Load portfolio data
    try:
        with open('portfolio/optimization_results.json', 'r') as f:
            portfolio_data = json.load(f)
    except:
        portfolio_data = None
    
    # Create visualizations
    perf_fig = create_performance_dashboard(performance_data)
    training_fig = create_training_history_chart(performance_data)
    pred_fig = create_prediction_timeline(predictions_data)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BTC Daily Prediction Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            /* Add your existing styles from create_dashboard_html() here */
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; line-height: 1.6; min-height: 100vh; padding: 20px; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #eee; }}
            .header h1 {{ color: #2c3e50; font-size: 2.5em; margin-bottom: 10px; }}
            .header p {{ color: #7f8c8d; font-size: 1.1em; }}
            /* ... rest of your styles ... */
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-rocket"></i> AI Financial Analytics Dashboard</h1>
                <p>Riverside Data Solutions, LLC & DeepSeek-powered Portfolio Optimization & BTC Predictions</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <!-- Portfolio Section -->
            {f'''
            <div class="card" style="margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h2><i class="fas fa-chart-pie"></i> Portfolio Optimization</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
                    <div class="metric">
                        <div class="label">Optimal Volatility</div>
                        <div class="value">{portfolio_data['performance']['optimal_volatility']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="label">Expected Return</div>
                        <div class="value">{portfolio_data['performance']['optimal_return']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="label">Sharpe Ratio</div>
                        <div class="value">{portfolio_data['performance']['optimal_sharpe']:.3f}</div>
                    </div>
                </div>
            </div>
            ''' if portfolio_data else ''}
            
            <!-- Current BTC Price -->
            <div style="text-align: center; background: #2c3e50; color: white; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h2>Current BTC Price: ${predictions_data['current_price']:,.2f if predictions_data else 'N/A'}</h2>
            </div>
            
            <!-- Predictions Section -->
            {f'''
            <div class="prediction-highlights" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2><i class="fas fa-bitcoin"></i> BTC 7-Day Predictions</h2>
                <p>Model: {predictions_data['model_used']} | Profit Score: {predictions_data['model_performance']['profit_score']:.4f} | Accuracy: {predictions_data['model_performance']['accuracy']*100:.1f}%</p>
                
                <!-- Prediction Visualization PNG -->
                <div style="text-align: center; margin: 20px 0;">
                    <img src="predictions/prediction_visualization.png" alt="BTC Prediction Chart" style="max-width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin: 20px 0;">
                    {''.join([f'''
                    <div style="background: {'#27ae60' if p['predicted_direction'] == 'UP' else '#e74c3c'}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="font-weight: bold; font-size: 1.2em;">Day {p['day']}</div>
                        <div style="font-size: 0.9em; opacity: 0.9;">{p['date']}</div>
                        <div style="font-weight: bold; margin: 5px 0;">${p['predicted_price']:,.2f}</div>
                        <div style="font-weight: bold;">
                            {p['predicted_return']:+.2f}%
                        </div>
                        <div style="font-size: 0.8em; margin-top: 5px;">{p['confidence']}</div>
                    </div>
                    ''' for p in predictions_data['predictions']])}
                </div>
                
                <div class="recommendation" style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50;">
                    <h3>Trading Recommendation: {predictions_data['trading_recommendation']['recommendation']}</h3>
                    <p><strong>Reasoning:</strong> {predictions_data['trading_recommendation']['reasoning']}</p>
                    <p><strong>Total 7-Day Return:</strong> {predictions_data['trading_recommendation']['total_return']:+.2f}%</p>
                </div>
            </div>
            ''' if predictions_data else '<p>No prediction data available</p>'}
            
            <!-- Interactive Charts -->
            <div class="chart" style="margin: 30px 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;">
                <div id="performance-chart"></div>
            </div>
            
            <div class="chart" style="margin: 30px 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;">
                <div id="training-chart"></div>
            </div>
            
            <!-- Add the interactive prediction chart if available -->
            {f'<div class="chart" style="margin: 30px 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;"><div id="prediction-chart"></div></div>' if predictions_data and pred_fig else ''}
        </div>
        
        <!-- Your existing JavaScript for Plotly charts -->
        <script>
            // Add your Plotly chart code here from the existing report_generation.py
        </script>
    </body>
    </html>
    """
    
    # Save HTML report
    os.makedirs('reports', exist_ok=True)
    with open('reports/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Also create index.html for GitHub Pages
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Interactive dashboard generated: reports/dashboard.html and index.html")
    
    # Add predictions section if available
    if predictions_data and pred_fig:
        up_count = predictions_data['summary']['bullish_days']
        down_count = predictions_data['summary']['bearish_days']
        total_return = predictions_data['summary']['total_7day_return']
        recommendation = predictions_data['trading_recommendation']
        
        html_content += f"""
            <div class="prediction-highlights">
                <h2>Next 7 Days Predictions</h2>
                <p><strong>UP Days:</strong> {up_count} | <strong>DOWN Days:</strong> {down_count}</p>
                <p><strong>Total 7-Day Return:</strong> {total_return:+.2f}%</p>
                <p><strong>Average Daily Return:</strong> {predictions_data['summary']['average_daily_return']:+.2f}%</p>
            </div>
            
            <div class="recommendation">
                <h3>Trading Recommendation: {recommendation['recommendation']}</h3>
                <p><strong>Confidence:</strong> {recommendation['confidence']}</p>
                <p><strong>Reasoning:</strong> {recommendation['reasoning']}</p>
                <p><strong>Stop Loss:</strong> ${recommendation['stop_loss']:.0f} | <strong>Take Profit:</strong> ${recommendation['take_profit']:.0f}</p>
            </div>
            
            <div class="chart">
                <div id="prediction-chart"></div>
            </div>
        """
    
    html_content += f"""
            <div class="chart">
                <div id="performance-chart"></div>
            </div>
            
            <div class="chart">
                <div id="training-chart"></div>
            </div>
        </div>
        
        <script>
    """
    
    # Add Plotly charts
    html_content += f"var perfChart = {perf_fig.to_json()};"
    html_content += f"Plotly.newPlot('performance-chart', perfChart.data, perfChart.layout);"
    
    html_content += f"var trainingChart = {training_fig.to_json()};"
    html_content += f"Plotly.newPlot('training-chart', trainingChart.data, trainingChart.layout);"
    
    if predictions_data and pred_fig:
        html_content += f"var predChart = {pred_fig.to_json()};"
        html_content += f"Plotly.newPlot('prediction-chart', predChart.data, predChart.layout);"
    
    html_content += """
        </script>
    </body>
    </html>
    """
    
    # Save HTML report with UTF-8 encoding
    os.makedirs('reports', exist_ok=True)
    with open('reports/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("HTML report generated: reports/dashboard.html")

def generate_markdown_report():
    """Generate markdown report for GitHub"""
    print("Generating markdown report...")
    
    performance_data, summary_data = load_performance_data()
    predictions_data = load_predictions()
    
    # Find best model
    best_model = max(performance_data.items(), key=lambda x: x[1]['test_profit_score'])
    
    markdown_content = f"""# BTC Daily Prediction Report

**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

- **Best Model**: {best_model[0]}
- **Best Profit Score**: {best_model[1]['test_profit_score']:.3f}
- **Best Accuracy**: {best_model[1]['test_accuracy']:.3f}
- **Total Training Samples**: {summary_data.get('total_samples', 0)}
- **Models Trained**: {summary_data.get('models_trained', 0)}

## Model Performance Comparison

| Model | Accuracy | F1-Score | Profit Score | Precision | Recall |
|-------|----------|----------|--------------|-----------|--------|
"""
    
    # Add model performance table
    for model, metrics in performance_data.items():
        markdown_content += f"| {model} | {metrics['test_accuracy']:.3f} | {metrics['test_f1']:.3f} | {metrics['test_profit_score']:.3f} | {metrics['test_precision']:.3f} | {metrics['test_recall']:.3f} |\n"
    
    # Add predictions section
    if predictions_data:
        recommendation = predictions_data['trading_recommendation']
        
        markdown_content += f"""

## Next 7 Days Predictions

### Trading Recommendation: {recommendation['recommendation']}
- **Confidence**: {recommendation['confidence']}
- **Reasoning**: {recommendation['reasoning']}
- **Total 7-Day Return**: {recommendation['total_return']:+.2f}%
- **Stop Loss**: ${recommendation['stop_loss']:.0f}
- **Take Profit**: ${recommendation['take_profit']:.0f}

### Daily Predictions:
| Date | Prediction | Return | Price |
|------|------------|--------|-------|
"""
        
        for pred in predictions_data['predictions']:
            markdown_content += f"| {pred['date']} | {pred['direction']} | {pred['predicted_return']:+.2f}% | ${pred['predicted_price']:.2f} |\n"
        
        up_count = predictions_data['summary']['bullish_days']
        markdown_content += f"\n**Summary**: {up_count} UP days, {7-up_count} DOWN days\n"
    
    markdown_content += f"""

## Files

- `models/best_model_info.json` - Best model configuration
- `reports/model_performance.json` - Detailed performance metrics
- `reports/dashboard.html` - Interactive dashboard
- `predictions/latest_predictions.json` - Latest predictions
- `predictions/prediction_visualization.png` - Prediction chart

![Prediction Visualization](predictions/prediction_visualization.png)

---

*Report automatically generated by BTC Prediction System*
"""
    
    # Save markdown report with UTF-8 encoding
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print("Markdown report generated: README.md")

def main():
    """Generate all reports"""
    try:
        print("Generating BTC Prediction Reports")
        print("="*40)
        
        generate_html_report()
        generate_markdown_report()
        
        print("\nAll reports generated successfully!")
        print("View interactive dashboard: reports/dashboard.html")
        print("View summary report: README.md")
        
    except Exception as e:
        print(f"Report generation failed: {e}")
        raise

if __name__ == "__main__":
    main()

    
