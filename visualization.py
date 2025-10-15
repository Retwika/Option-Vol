"""
Visualization functions for option strategies and Greeks.

This module contains all plotting functions using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from option_strategies import OptionStrategy


def create_payoff_chart(prices: np.ndarray, payoff: np.ndarray, title: str) -> go.Figure:
    """
    Create a payoff chart.
    
    Args:
        prices: Array of underlying prices
        payoff: Array of payoff values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices, 
        y=payoff, 
        mode='lines', 
        name='Payoff',
        line=dict(width=3)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Underlying Price",
        yaxis_title="Payoff",
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def create_greeks_chart(prices: np.ndarray, greek_values: np.ndarray, 
                       greek_name: str, title: str, currency_symbol: str = "$") -> go.Figure:
    """
    Create a Greek chart.
    
    Args:
        prices: Array of underlying prices
        greek_values: Array of Greek values
        greek_name: Name of the Greek (Delta, Gamma, etc.)
        title: Chart title
        currency_symbol: Currency symbol to display ($ or ₹)
        
    Returns:
        Plotly figure
    """
    # Handle edge cases for empty or invalid data
    if len(greek_values) == 0 or np.all(np.isnan(greek_values)):
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, template="plotly_white")
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices, 
        y=greek_values, 
        mode='lines', 
        name=greek_name,
        line=dict(width=3),
        hovertemplate=f'<b>Price</b>: {currency_symbol}%{{x:.2f}}<br><b>{greek_name}</b>: %{{y:.6f}}<extra></extra>'
    ))
    
    # Add zero line for all Greeks except when using log scale
    if greek_name.lower() not in ['gamma', 'vega'] or not (np.any(np.abs(greek_values) > 0)):
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Use log scale for Gamma and Vega if they have significant absolute values
    # but only if all values have the same sign to avoid log scale issues
    use_log = False
    if greek_name in ["Gamma", "Vega"]:
        abs_values = np.abs(greek_values)
        if np.any(abs_values > 0):
            # Check if all values have the same sign (all positive or all negative)
            all_positive = np.all(greek_values >= 0)
            all_negative = np.all(greek_values <= 0)
            max_abs_value = np.max(abs_values)
            min_abs_value = np.min(abs_values[abs_values > 0]) if np.any(abs_values > 0) else 1
            
            # Use log scale if range is large and all same sign
            if (all_positive or all_negative) and max_abs_value / min_abs_value > 10:
                use_log = True
    
    yaxis_type = "log" if use_log else "linear"
    
    fig.update_layout(
        title=title,
        xaxis_title="Underlying Price",
        yaxis_title=greek_name,
        yaxis_type=yaxis_type,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def create_combined_greeks_charts(prices: np.ndarray, delta: np.ndarray, 
                                 gamma: np.ndarray, theta: np.ndarray, 
                                 vega: np.ndarray, currency_symbol: str = "$") -> List[go.Figure]:
    """
    Create charts for all Greeks.
    
    Args:
        prices: Array of underlying prices
        delta: Array of delta values
        gamma: Array of gamma values
        theta: Array of theta values
        vega: Array of vega values
        currency_symbol: Currency symbol to display ($ or ₹)
        
    Returns:
        List of Plotly figures for each Greek
    """
    greeks_data = [
        (delta, "Delta", "Delta"),
        (gamma, "Gamma", "Gamma"),
        (theta, "Theta", "Theta"),
        (vega, "Vega", "Vega")
    ]
    
    figures = []
    for values, name, title in greeks_data:
        fig = create_greeks_chart(prices, values, name, f"{title} - Strategy", currency_symbol)
        figures.append(fig)
    
    return figures


def create_theta_surface(prices: np.ndarray, time_grid: np.ndarray, 
                        theta_surface: np.ndarray, title: str, currency_symbol: str = "$") -> go.Figure:
    """
    Create a 3D theta surface plot.
    
    Args:
        prices: Array of underlying prices
        time_grid: Array of time values (in days)
        theta_surface: 2D array of theta values
        title: Chart title
        currency_symbol: Currency symbol to display ($ or ₹)
        
    Returns:
        Plotly 3D surface figure
    """
    fig = go.Figure(data=[go.Surface(
        z=theta_surface,
        x=prices,
        y=time_grid,
        colorscale='RdBu',
        colorbar=dict(title="Theta"),
        hovertemplate=f'<b>Price</b>: {currency_symbol}%{{x:.2f}}<br>' +
                      '<b>Days to Expiry</b>: %{y:.0f}<br>' +
                      '<b>Theta</b>: %{z:.4f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Underlying Price',
            yaxis_title='Days to Expiry',
            zaxis_title='Theta'
        ),
        height=600,
        template="plotly_white"
    )
    
    return fig


def create_strategy_comparison_chart(strategy: OptionStrategy, prices: np.ndarray, 
                                   T: float, r: float, sigma: float, currency_symbol: str = "$") -> go.Figure:
    """
    Create a chart comparing individual legs and total strategy payoff.
    
    Args:
        strategy: OptionStrategy object
        prices: Array of underlying prices
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        currency_symbol: Currency symbol to display ($ or ₹)
        
    Returns:
        Plotly figure with multiple traces
    """
    fig = go.Figure()
    
    # Add individual legs
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, leg in enumerate(strategy.legs):
        leg_payoff = leg.calculate_payoff(prices)
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=prices,
            y=leg_payoff,
            mode='lines',
            name=f'Leg {i+1}: {leg}',
            line=dict(width=2, dash='dash', color=color),
            opacity=0.7
        ))
    
    # Add total strategy
    total_payoff = strategy.calculate_total_payoff(prices)
    fig.add_trace(go.Scatter(
        x=prices,
        y=total_payoff,
        mode='lines',
        name='Total Strategy',
        line=dict(width=4, color='white')
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Strategy Breakdown: Individual Legs vs Total",
        xaxis_title="Underlying Price",
        yaxis_title="Payoff",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_profit_loss_zones_chart(prices: np.ndarray, payoff: np.ndarray, currency_symbol: str = "$") -> go.Figure:
    """
    Create a payoff chart with profit/loss zones highlighted.
    
    Args:
        prices: Array of underlying prices
        payoff: Array of payoff values
        currency_symbol: Currency symbol to display ($ or ₹)
        
    Returns:
        Plotly figure with colored profit/loss zones
    """
    fig = go.Figure()
    
    # Find breakeven points
    breakeven_indices = []
    for i in range(len(payoff) - 1):
        if payoff[i] * payoff[i + 1] < 0:  # Sign change indicates breakeven
            breakeven_indices.append(i)
    
    # Add profit zones (green) and loss zones (red)
    profit_mask = payoff > 0
    loss_mask = payoff < 0
    
    if np.any(profit_mask):
        fig.add_trace(go.Scatter(
            x=prices[profit_mask],
            y=payoff[profit_mask],
            mode='lines',
            name='Profit Zone',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
    
    if np.any(loss_mask):
        fig.add_trace(go.Scatter(
            x=prices[loss_mask],
            y=payoff[loss_mask],
            mode='lines',
            name='Loss Zone',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
    
    # Add breakeven points
    for idx in breakeven_indices:
        breakeven_price = prices[idx]
        fig.add_vline(
            x=breakeven_price,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Breakeven: {breakeven_price:.2f}",
            annotation_position="top"
        )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Profit/Loss Analysis",
        xaxis_title="Underlying Price",
        yaxis_title="Payoff",
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def create_risk_metrics_table(strategy: OptionStrategy, current_price: float,
                             T: float, r: float, sigma: float, currency_symbol: str = "$") -> go.Figure:
    """
    Create a table showing key risk metrics for the strategy.
    
    Args:
        strategy: OptionStrategy object
        current_price: Current underlying price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        currency_symbol: Currency symbol to display ($ or ₹)
        
    Returns:
        Plotly table figure
    """
    # Calculate metrics at current price
    current_prices = np.array([current_price])
    total_payoff = strategy.calculate_total_payoff(current_prices)[0]
    delta, gamma, theta, vega = strategy.calculate_total_greeks(
        current_prices, T, r, sigma
    )
    
    # Calculate some additional metrics
    price_range = strategy.get_price_range()
    total_payoff_range = strategy.calculate_total_payoff(price_range)
    max_profit = np.max(total_payoff_range) if np.max(total_payoff_range) != np.inf else "Unlimited"
    max_loss = np.min(total_payoff_range) if np.min(total_payoff_range) != -np.inf else "Unlimited"
    
    # Create table data
    metrics = [
        "Current P&L",
        "Delta",
        "Gamma", 
        "Theta (per day)",
        "Vega (per 1% vol)",
        "Max Profit",
        "Max Loss"
    ]
    
    values = [
        f"{currency_symbol}{total_payoff:.2f}",
        f"{delta[0]:.4f}",
        f"{gamma[0]:.6f}",
        f"{currency_symbol}{theta[0]:.4f}",
        f"{currency_symbol}{vega[0]:.4f}",
        f"{currency_symbol}{max_profit:.2f}" if isinstance(max_profit, (int, float)) else str(max_profit),
        f"{currency_symbol}{max_loss:.2f}" if isinstance(max_loss, (int, float)) else str(max_loss)
    ]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Risk Metric</b>', '<b>Value</b>'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[metrics, values],
            fill_color='white',
            align='left',
            font=dict(size=12, color='black')
        )
    )])
    
    fig.update_layout(
        title=f"Risk Metrics @ {currency_symbol}{current_price}",
        height=300
    )
    
    return fig
