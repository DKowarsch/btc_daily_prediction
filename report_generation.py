# src/report_generation.py 
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

def create_performance_dashboard(performance_data):
    """Create model performance comparison dashboard"""
    models = list(performance_data.keys())
    metrics = ['test_accuracy', 'test_f1', 'test_profit_score', 'test_precision', 'test_recall']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Accuracy', 'F1-Score', 'Profit Score', 'Precision', 'Recall', 'Model Parameters'],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Professional color palette
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    # Add bars for each metric with better colors
    for i, metric in enumerate(metrics):
        values = [performance_data[model][metric] for model in models]
        row = i // 3 + 1
        col = i % 3 + 1
        
        fig.add_trace(
            go.Bar(name=metric, x=models, y=values, 
                   text=[f'{v:.3f}' for v in values], textposition='auto',
                   marker_color=colors[i], opacity=0.8),
            row=row, col=col
        )
    
    # Add parameters metric
    parameters = [performance_data[model]['parameters'] for model in models]
    fig.add_trace(
        go.Bar(name='Parameters', x=models, y=parameters,
               text=[f'{v:,}' for v in parameters], textposition='auto',
               marker_color='#95a5a6', opacity=0.8),
        row=2, col=3
    )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=600,
        showlegend=False,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white'
    )
    
    # Update axes for better readability
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickformat=',.0%', row=1, col=1)
    fig.update_yaxes(tickformat=',.0%', row=1, col=2)
    fig.update_yaxes(tickformat=',.0%', row=1, col=3)
    fig.update_yaxes(tickformat=',.0%', row=2, col=1)
    fig.update_yaxes(tickformat=',.0%', row=2, col=2)
    
    return fig

def create_prediction_timeline(predictions_data):
    """Create prediction timeline visualization with actual dates"""
    if not predictions_data:
        return None
        
    predictions = predictions_data['predictions']
    dates = [p['date'] for p in predictions]
    returns = [p['predicted_return'] for p in predictions]
    prices = [p['predicted_price'] for p in predictions]
    directions = [p['predicted_direction'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Predicted Returns', 'Predicted Prices'],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15
    )
    
    # Returns chart with gradient colors
    colors = []
    for ret in returns:
        if ret > 0:
            # Green gradient based on return magnitude
            intensity = min(abs(ret) / 5, 1)  # Normalize to 0-1
            colors.append(f'rgba(46, 204, 113, {0.3 + intensity*0.7})')
        else:
            # Red gradient based on return magnitude
            intensity = min(abs(ret) / 5, 1)
            colors.append(f'rgba(231, 76, 60, {0.3 + intensity*0.7})')
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=returns,
            name='Return',
            marker_color=colors,
            text=[f'{r:+.2f}%' for r in returns],
            textposition='auto',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Return: %{text}<br>" +
                "Direction: %{customdata[0]}<br>" +
                "Confidence: %{customdata[1]}<br>" +
                "<extra></extra>"
            ),
            customdata=list(zip(directions, confidences))
        ),
        row=1, col=1
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10, color='#2980b9'),
            text=[f'${p:,.2f}' for p in prices],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Price: %{text}<br>" +
                "<extra></extra>"
            )
        ),
        row=2, col=1
    )
    
    # Add current price reference line
    current_price = predictions_data['current_price']
    fig.add_hline(
        y=current_price, 
        line_dash="dash", 
        line_color="#7f8c8d", 
        annotation_text=f"Current: ${current_price:,.2f}",
        annotation_position="top left",
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'BTC 7-Day Price Predictions',
        height=600,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Price ($)', row=2, col=1)
    fig.update_yaxes(tickprefix='$', row=2, col=1)
    
    return fig

def create_training_history_chart(performance_data):
    """Create training history and parameter comparison"""
    models = list(performance_data.keys())
    parameters = [performance_data[model]['parameters'] for model in models]
    accuracy = [performance_data[model]['test_accuracy'] for model in models]
    profit_scores = [performance_data[model]['test_profit_score'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Efficiency Analysis', 'Performance Distribution'],
        specs=[[{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Model size vs accuracy - bubble chart
    sizes = [p/1000 for p in parameters]  # Scale for bubble size
    
    fig.add_trace(
        go.Scatter(
            x=parameters,
            y=accuracy,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=profit_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Profit Score", x=0.46)
            ),
            text=models,
            textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Parameters: %{x:,}<br>" +
                "Accuracy: %{y:.3f}<br>" +
                "Profit Score: %{marker.color:.3f}<br>" +
                "<extra></extra>"
            )
        ),
        row=1, col=1
    )
    
    # Performance distribution with better colors
    fig.add_trace(
        go.Pie(
            labels=models,
            values=profit_scores,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3),
            hovertemplate="<b>%{label}</b><br>Profit Score: %{value:.3f}<br>Share: %{percent}",
            textinfo='percent+label'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Model Analysis Dashboard',
        height=450,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    fig.update_xaxes(title_text='Number of Parameters', row=1, col=1)
    fig.update_yaxes(title_text='Test Accuracy', tickformat=',.0%', row=1, col=1)
    
    return fig

def generate_html_report():
    """Generate complete HTML report with professional design"""
    print("Generating professional performance report...")
    
    # Load data
    performance_data, summary_data = load_performance_data()
    predictions_data = load_predictions()
    portfolio_data = load_portfolio_data()
    
    # Create visualizations
    perf_fig = create_performance_dashboard(performance_data)
    training_fig = create_training_history_chart(performance_data)
    pred_fig = create_prediction_timeline(predictions_data)
    
    # Get best model info
    best_model = max(performance_data.items(), key=lambda x: x[1]['test_profit_score'])
    best_model_name = best_model[0]
    best_model_score = best_model[1]['test_profit_score']
    
    # Generate professional HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Financial Analytics Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 30px;
                border-bottom: 2px solid #f0f0f0;
            }}
            
            .header h1 {{
                color: #2c3e50;
                font-size: 2.8em;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            
            .header .subtitle {{
                color: #7f8c8d;
                font-size: 1.2em;
                margin-bottom: 20px;
            }}
            
            .timestamp {{
                background: #f8f9fa;
                padding: 12px 20px;
                border-radius: 10px;
                display: inline-block;
                color: #666;
                font-size: 0.9em;
            }}
            
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 30px;
                margin-bottom: 40px;
            }}
            
            .card {{
                background: #fff;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.08);
                border: 1px solid #f0f0f0;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.12);
            }}
            
            .card h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 1.4em;
                font-weight: 600;
            }}
            
            .card h2 i {{
                color: #667eea;
            }}
            
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin: 20px 0;
            }}
            
            .metric {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                transition: transform 0.2s ease;
            }}
            
            .metric:hover {{
                transform: scale(1.05);
            }}
            
            .metric .value {{
                font-size: 1.8em;
                font-weight: 700;
                margin: 5px 0;
            }}
            
            .metric .label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            
            .prediction-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
            }}
            
            .prediction-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 15px;
                margin: 25px 0;
            }}
            
            .prediction-day {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
                border: 2px solid transparent;
            }}
            
            .prediction-day.up {{
                border-color: #2ecc71;
                background: linear-gradient(135deg, #ffffff 0%, #d5f4e6 100%);
            }}
            
            .prediction-day.down {{
                border-color: #e74c3c;
                background: linear-gradient(135deg, #ffffff 0%, #fadbd8 100%);
            }}
            
            .prediction-day:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }}
            
            .recommendation {{
                background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
                padding: 25px;
                border-radius: 12px;
                margin: 30px 0;
                border-left: 5px solid #28a745;
            }}
            
            .recommendation h3 {{
                color: #155724;
                margin-bottom: 15px;
                font-size: 1.3em;
            }}
            
            .chart-container {{
                margin: 25px 0;
                padding: 20px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #f0f0f0;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            
            .badge {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                margin-left: 10px;
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
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 20px;
                }}
                
                .dashboard-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .metric-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-chart-line"></i> AI Financial Analytics Dashboard</h1>
                <p class="subtitle">Riverside Data Solutions, LLC • DeepSeek AI-Powered Predictions</p>
                <div class="timestamp">
                    <i class="fas fa-sync-alt"></i> Last updated: {datetime.now().strftime('%Y-%m-%d %H
