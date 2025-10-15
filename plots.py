"""
Plotting utilities for volatility analysis.
Centralized plotting functions to eliminate redundancy and ensure consistency.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_utils import convert_iv_for_display
from config import PLOT_SETTINGS, DISPLAY_SETTINGS


def create_3d_volatility_surface(surface_df, iv_column='VW_IV', title_prefix=""):
    """
    Create 3D volatility surface plot with consistent formatting.
    
    Args:
        surface_df: DataFrame with columns ['Strike', 'Expiry', iv_column]
        iv_column: Column name for implied volatility data
        title_prefix: Optional prefix for plot title
    """
    try:
        if surface_df.empty:
            st.warning("No data available for 3D surface.")
            return None
            
        today = pd.Timestamp.now().normalize()
        surface_df_copy = surface_df.copy()
        
        # Parse expiry dates and calculate time to expiry
        try:
            # Try NIFTY format first
            surface_df_copy['TimeToExpiry'] = (pd.to_datetime(surface_df_copy['Expiry'], format='%d-%b-%Y') - today).dt.days
        except ValueError:
            try:
                # Try ISO format for US options
                surface_df_copy['TimeToExpiry'] = (pd.to_datetime(surface_df_copy['Expiry'], format='%Y-%m-%d') - today).dt.days
            except ValueError:
                # Use pandas auto-detection as fallback
                surface_df_copy['TimeToExpiry'] = (pd.to_datetime(surface_df_copy['Expiry']) - today).dt.days
        
        # Remove negative/expired options and filter for reasonable data - include today's expiry
        surface_df_copy = surface_df_copy[surface_df_copy['TimeToExpiry'] >= 0]
        surface_df_copy = surface_df_copy.dropna(subset=[iv_column, 'Strike', 'TimeToExpiry'])
        
        if surface_df_copy.empty:
            st.warning("No valid data available for 3D surface after filtering.")
            return None
        
        # Get unique strikes and times, sorted
        strikes = sorted(surface_df_copy['Strike'].unique())
        times = sorted(surface_df_copy['TimeToExpiry'].unique())
        
        if len(strikes) < 2 or len(times) < 2:
            st.warning("Insufficient data points for 3D surface (need at least 2 strikes and 2 expiries).")
            return None
        
        # Create meshgrid and interpolate data
        strike_grid, time_grid = np.meshgrid(strikes, times)
        
        # Interpolate IV values - create a grid
        iv_grid = np.full(strike_grid.shape, np.nan)
        
        for i, time_val in enumerate(times):
            for j, strike_val in enumerate(strikes):
                matching_row = surface_df_copy[
                    (surface_df_copy['TimeToExpiry'] == time_val) & 
                    (surface_df_copy['Strike'] == strike_val)
                ]
                if not matching_row.empty:
                    # Convert to display format (percentage)
                    iv_grid[i, j] = convert_iv_for_display(matching_row[iv_column].iloc[0])
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=iv_grid,
            x=strike_grid,
            y=time_grid,
            colorscale=PLOT_SETTINGS['surface_colorscale'],
            name='Volatility Surface',
            hovertemplate='<b>Strike:</b> %{x}<br><b>Days:</b> %{y}<br><b>IV:</b> %{z:.2f}%<extra></extra>'
        )])
        
        title = f"{title_prefix}3D Volatility Surface" if title_prefix else "3D Volatility Surface"
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility (%)',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
            ),
            height=PLOT_SETTINGS['surface_height'],
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D surface: {str(e)}")
        return None


def create_iv_heatmap(surface_df, iv_column='VW_IV'):
    """Create IV heatmap with consistent formatting."""
    try:
        if surface_df.empty:
            st.warning("No data available for heatmap.")
            return None
            
        # Prepare data for heatmap
        heatmap_data = surface_df.pivot_table(
            values=iv_column, 
            index='Expiry', 
            columns='Strike', 
            aggfunc='mean'
        )
        
        if heatmap_data.empty:
            st.warning("No data available for heatmap after pivot.")
            return None
        
        # Convert to display format (percentage)
        heatmap_data_display = heatmap_data.applymap(
            lambda x: convert_iv_for_display(x) if pd.notnull(x) else x
        )
        
        fig = px.imshow(
            heatmap_data_display,
            labels=dict(x="Strike Price", y="Expiry", color="IV (%)"),
            color_continuous_scale=PLOT_SETTINGS['heatmap_colorscale'],
            title="Implied Volatility Heatmap"
        )
        
        fig.update_layout(
            height=PLOT_SETTINGS['heatmap_height'],
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None


def create_volatility_smile_plot(vol_surface, expiry, current_price, iv_column='VW_IV'):
    """Create volatility smile plot for a specific expiry."""
    try:
        expiry_data = vol_surface[vol_surface['Expiry'] == expiry].copy()
        
        if expiry_data.empty:
            return None
            
        expiry_data = expiry_data.sort_values('Strike')
        
        # Calculate moneyness
        expiry_data['Moneyness'] = expiry_data['Strike'] / current_price
        
        fig = go.Figure()
        
        # Market IV
        fig.add_trace(go.Scatter(
            x=expiry_data['Strike'],
            y=convert_iv_for_display(expiry_data[iv_column]),
            mode='markers+lines',
            name='Market IV',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Market IV</b><br>Strike: %{x}<br>IV: %{y:.2f}%<br>Moneyness: %{customdata:.3f}<extra></extra>',
            customdata=expiry_data['Moneyness']
        ))
        
        # Add ATM line
        fig.add_vline(
            x=current_price, 
            line_dash="dash", 
            line_color="red",
            annotation_text="ATM"
        )
        
        fig.update_layout(
            title=f"Volatility Smile - {expiry}",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating smile plot: {str(e)}")
        return None


def create_mispricing_scatter(mispricing_opportunities, max_display=50):
    """Create scatter plot of mispricing opportunities."""
    try:
        if not mispricing_opportunities:
            return None
            
        # Limit display for performance
        display_data = mispricing_opportunities[:max_display]
        
        df = pd.DataFrame(display_data)
        
        fig = px.scatter(
            df,
            x='Strike',
            y='IV_Spread_Pct',
            color='Expiry',
            size='Total_Volume',
            hover_data=['Option_Type', 'Market_Price', 'Theoretical_Price'],
            title=f"Top {len(display_data)} Mispricing Opportunities",
            labels={'IV_Spread_Pct': 'IV Spread (%)', 'Strike': 'Strike Price'}
        )
        
        fig.update_layout(
            height=500,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating mispricing scatter: {str(e)}")
        return None


def display_plot_with_fallback(plot_func, *args, **kwargs):
    """
    Display a plot with graceful fallback on error.
    
    Args:
        plot_func: Function that returns a plotly figure
        *args, **kwargs: Arguments to pass to plot_func
    """
    try:
        fig = plot_func(*args, **kwargs)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            return True
        else:
            st.info("No data available for this visualization.")
            return False
    except Exception as e:
        st.error(f"Failed to create plot: {str(e)}")
        return False
