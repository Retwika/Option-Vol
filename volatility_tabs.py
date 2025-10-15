"""
Modular volatility surface analysis for the Streamlit app.
Refactored from monolithic 1200+ line file into maintainable components.

Key improvements:
- Eliminated redundant 3D surface builders and duplicate code
- Consistent IV unit handling (percentage display)
- Cached volatility surface calculations
- Modular plotting and analysis components
- Clear separation of data processing, modeling, and visualization
- Fixed butterfly arbitrage math for unequal spacing
- Added robustness checks and tolerance for noise
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
from scipy.interpolate import griddata
from datetime import datetime

# Core imports
from data_utils import convert_iv_for_display
from nifty_data import (
    fetch_nifty_options, process_nifty_options_data,
    fetch_us_options, process_us_options_data,
    calculate_vol_surface_metrics
)
from config import DISPLAY_SETTINGS, PLOT_SETTINGS


def identify_arbitrage_opportunities(options_data: pd.DataFrame, max_opportunities: int = 20) -> List[Dict]:
    """
    Identify arbitrage opportunities using standard no-arbitrage conditions.
    
    Checks for:
    1. Calendar arbitrage: C(K,T2) < C(K,T1) for T2 > T1
    2. Butterfly arbitrage: Convexity violation
    3. Vertical arbitrage: C(K1) < C(K2) for K1 < K2
    """
    if options_data is None or options_data.empty:
        return []
    
    arbitrage_opportunities = []
    
    try:
        # Prepare data - work with call options only for arbitrage checks
        df = options_data[options_data['Type'] == 'CE'].copy()
        df = df[df['Last_Price'] > 0].copy()  # Filter out zero prices
        
        # Convert expiry to datetime for sorting
        today = pd.Timestamp.now().normalize()
        try:
            df['ExpiryDate'] = pd.to_datetime(df['Expiry'], format='%d-%b-%Y')
        except:
            try:
                df['ExpiryDate'] = pd.to_datetime(df['Expiry'], format='%Y-%m-%d')
            except:
                df['ExpiryDate'] = pd.to_datetime(df['Expiry'])
        
        df['DaysToExpiry'] = (df['ExpiryDate'] - today).dt.days
        df = df[df['DaysToExpiry'] >= 0].copy()
        
        # 1. Calendar Arbitrage Check
        calendar_arbs = _check_calendar_arbitrage(df)
        arbitrage_opportunities.extend(calendar_arbs)
        
        # 2. Butterfly Arbitrage Check  
        butterfly_arbs = _check_butterfly_arbitrage(df)
        arbitrage_opportunities.extend(butterfly_arbs)
        
        # 3. Vertical Arbitrage Check
        vertical_arbs = _check_vertical_arbitrage(df)
        arbitrage_opportunities.extend(vertical_arbs)
        
        # Sort by profit potential and limit results
        arbitrage_opportunities.sort(key=lambda x: x.get('Profit_Potential', 0), reverse=True)
        
        return arbitrage_opportunities[:max_opportunities]
        
    except Exception as e:
        print(f"Error in arbitrage detection: {str(e)}")
        return []


def _check_calendar_arbitrage(df: pd.DataFrame) -> List[Dict]:
    """Check for calendar arbitrage: C(K,T2) < C(K,T1) for T2 > T1"""
    arbitrages = []
    
    # Group by strike
    for strike in df['Strike'].unique():
        strike_data = df[df['Strike'] == strike].sort_values('DaysToExpiry')
        
        if len(strike_data) < 2:
            continue
            
        # Check each pair of options with different expiries
        for i in range(len(strike_data)):
            for j in range(i + 1, len(strike_data)):
                short_exp = strike_data.iloc[i]
                long_exp = strike_data.iloc[j]
                
                # Safe price extraction with column existence check
                short_price = 0
                long_price = 0
                
                if 'Mid_Price' in short_exp and pd.notna(short_exp['Mid_Price']):
                    short_price = short_exp['Mid_Price']
                elif 'Last_Price' in short_exp and pd.notna(short_exp['Last_Price']):
                    short_price = short_exp['Last_Price']
                
                if 'Mid_Price' in long_exp and pd.notna(long_exp['Mid_Price']):
                    long_price = long_exp['Mid_Price']
                elif 'Last_Price' in long_exp and pd.notna(long_exp['Last_Price']):
                    long_price = long_exp['Last_Price']
                
                # Calendar arbitrage: longer expiry should cost more (with small tolerance)
                tolerance = 0.05  # Small tolerance for market noise
                if long_price < short_price - tolerance:
                    profit = short_price - long_price
                    
                    arbitrages.append({
                        'Type': 'Calendar Arbitrage',
                        'Strike': strike,
                        'Short_Expiry': short_exp['Expiry'],
                        'Long_Expiry': long_exp['Expiry'],
                        'Short_Price': short_price,
                        'Long_Price': long_price,
                        'Profit_Potential': profit,
                        'Strategy': f"Sell {short_exp['Expiry']} Call, Buy {long_exp['Expiry']} Call"
                    })
    
    return arbitrages


def _check_butterfly_arbitrage(df: pd.DataFrame) -> List[Dict]:
    """Check for butterfly arbitrage: Convexity violation in option prices"""
    arbitrages = []
    
    # Group by expiry
    for expiry in df['Expiry'].unique():
        exp_data = df[df['Expiry'] == expiry].sort_values('Strike')
        strikes = sorted(exp_data['Strike'].unique())
        
        # Check convexity for each possible butterfly (more flexible spacing)
        for i in range(1, len(strikes) - 1):
            k_low = strikes[i-1]
            k_mid = strikes[i]
            k_high = strikes[i+1]
            
            # Allow flexible strike spacing (not requiring exact equal spacing)
            spacing_low = k_mid - k_low
            spacing_high = k_high - k_mid
            
            # Only require reasonably similar spacing (within 50% difference)
            if abs(spacing_low - spacing_high) > 0.5 * max(spacing_low, spacing_high):
                continue
                
            try:
                # Safe price extraction with column existence check
                low_data = exp_data[exp_data['Strike'] == k_low]
                mid_data = exp_data[exp_data['Strike'] == k_mid]
                high_data = exp_data[exp_data['Strike'] == k_high]
                
                # Robust price extraction
                c_low = 0
                c_mid = 0
                c_high = 0
                
                if not low_data.empty:
                    if 'Mid_Price' in low_data.columns and pd.notna(low_data['Mid_Price'].iloc[0]):
                        c_low = low_data['Mid_Price'].iloc[0]
                    elif 'Last_Price' in low_data.columns and pd.notna(low_data['Last_Price'].iloc[0]):
                        c_low = low_data['Last_Price'].iloc[0]
                
                if not mid_data.empty:
                    if 'Mid_Price' in mid_data.columns and pd.notna(mid_data['Mid_Price'].iloc[0]):
                        c_mid = mid_data['Mid_Price'].iloc[0]
                    elif 'Last_Price' in mid_data.columns and pd.notna(mid_data['Last_Price'].iloc[0]):
                        c_mid = mid_data['Last_Price'].iloc[0]
                
                if not high_data.empty:
                    if 'Mid_Price' in high_data.columns and pd.notna(high_data['Mid_Price'].iloc[0]):
                        c_high = high_data['Mid_Price'].iloc[0]
                    elif 'Last_Price' in high_data.columns and pd.notna(high_data['Last_Price'].iloc[0]):
                        c_high = high_data['Last_Price'].iloc[0]
                
                # Skip if any price is zero or invalid
                if c_low <= 0 or c_mid <= 0 or c_high <= 0:
                    continue
                
                # Calculate weights for unequal spacing
                weight_low = spacing_high / (spacing_low + spacing_high)
                weight_high = spacing_low / (spacing_low + spacing_high)
                
                # Weighted convexity check for consistent math
                interpolated_mid = weight_low * c_low + weight_high * c_high
                convexity = interpolated_mid - c_mid
                
                # Dynamic tolerance based on option price level
                tolerance = max(0.5, 0.01 * c_mid)
                
                if convexity < -tolerance:
                    # Calculate cost using the same weights as convexity for consistency
                    butterfly_cost = weight_low * c_low - c_mid + weight_high * c_high
                    profit = -butterfly_cost  # Arbitrage profit is negative of cost
                    
                    # Strategy description with proper ratios
                    if spacing_low == spacing_high:
                        strategy = f"Buy {k_low} Call, Sell 2x {k_mid} Call, Buy {k_high} Call"
                    else:
                        strategy = f"Buy {weight_low:.2f}x {k_low} Call, Sell 1x {k_mid} Call, Buy {weight_high:.2f}x {k_high} Call"
                    
                    arbitrages.append({
                        'Type': 'Butterfly Arbitrage',
                        'Strike_Low': k_low,
                        'Strike_Mid': k_mid,
                        'Strike_High': k_high,
                        'Expiry': expiry,
                        'Price_Low': c_low,
                        'Price_Mid': c_mid,
                        'Price_High': c_high,
                        'Convexity_Violation': convexity,
                        'Butterfly_Cost': butterfly_cost,
                        'Profit_Potential': profit,
                        'Strategy': strategy,
                        'Spacing_Low': spacing_low,
                        'Spacing_High': spacing_high
                    })
            except (IndexError, KeyError, AttributeError):
                continue
    
    return arbitrages


def _check_vertical_arbitrage(df: pd.DataFrame) -> List[Dict]:
    """Check for vertical arbitrage: C(K1) < C(K2) for K1 < K2"""
    arbitrages = []
    
    # Group by expiry
    for expiry in df['Expiry'].unique():
        exp_data = df[df['Expiry'] == expiry].sort_values('Strike')
        
        if len(exp_data) < 2:
            continue
            
        # Check each adjacent pair of strikes
        for i in range(len(exp_data) - 1):
            lower_strike = exp_data.iloc[i]
            higher_strike = exp_data.iloc[i + 1]
            
            # Safe price extraction
            lower_price = 0
            higher_price = 0
            
            if 'Mid_Price' in lower_strike and pd.notna(lower_strike['Mid_Price']):
                lower_price = lower_strike['Mid_Price']
            elif 'Last_Price' in lower_strike and pd.notna(lower_strike['Last_Price']):
                lower_price = lower_strike['Last_Price']
            
            if 'Mid_Price' in higher_strike and pd.notna(higher_strike['Mid_Price']):
                higher_price = higher_strike['Mid_Price']
            elif 'Last_Price' in higher_strike and pd.notna(higher_strike['Last_Price']):
                higher_price = higher_strike['Last_Price']
            
            # Vertical arbitrage: lower strike should have higher price (with small tolerance)
            tolerance = 0.05  # Small tolerance for market noise
            if lower_price < higher_price - tolerance:
                profit = higher_price - lower_price
                
                arbitrages.append({
                    'Type': 'Vertical Arbitrage',
                    'Strike_Low': lower_strike['Strike'],
                    'Strike_High': higher_strike['Strike'],
                    'Expiry': expiry,
                    'Price_Low': lower_price,
                    'Price_High': higher_price,
                    'Profit_Potential': profit,
                    'Strategy': f"Buy {lower_strike['Strike']} Call, Sell {higher_strike['Strike']} Call"
                })
    
    return arbitrages


def create_3d_surface_plot(surface_df: pd.DataFrame, iv_column: str = 'VW_IV', title: str = "3D Volatility Surface") -> Optional[go.Figure]:
    """
    Create 3D volatility surface plot with consistent IV formatting.
    Unified implementation to replace redundant surface builders.
    """
    try:
        if surface_df.empty:
            return None
            
        today = pd.Timestamp.now().normalize()
        df = surface_df.copy()
        
        # Parse expiry dates consistently
        try:
            df['TimeToExpiry'] = (pd.to_datetime(df['Expiry'], format='%d-%b-%Y') - today).dt.days
        except ValueError:
            try:
                df['TimeToExpiry'] = (pd.to_datetime(df['Expiry'], format='%Y-%m-%d') - today).dt.days
            except ValueError:
                df['TimeToExpiry'] = (pd.to_datetime(df['Expiry']) - today).dt.days
        
        # Filter valid data - include today's expiry (>= 0 instead of > 0)
        df = df[(df['TimeToExpiry'] >= 0) & df[iv_column].notna()]
        
        if len(df) < PLOT_SETTINGS.get('min_surface_points', 4):
            st.warning(f"Insufficient data for 3D surface ({len(df)} points)")
            return None
        
        # Create grid
        strikes = sorted(df['Strike'].unique())
        times = sorted(df['TimeToExpiry'].unique())
        
        if len(strikes) < 2 or len(times) < 2:
            st.warning("Need at least 2 strikes and 2 expiries for 3D surface")
            return None
        
        # Create wireframe-style surface with all data points visible
        fig = go.Figure()
        
        # Add all data points as scatter3d (dots)
        fig.add_trace(go.Scatter3d(
            x=df['Strike'],
            y=df['TimeToExpiry'],
            z=convert_iv_for_display(df[iv_column]),
            mode='markers',
            marker=dict(
                size=4,
                color=convert_iv_for_display(df[iv_column]),
                colorscale=PLOT_SETTINGS.get('surface_colorscale', 'Viridis'),
                colorbar=dict(
                    title="IV (%)",
                    x=0.96,  # Position to the right
                    len=1,   # Shorter length
                    thickness=15,  # Thinner
                    title_side="bottom"
                ),
                opacity=0.8
            ),
            name='Market Data',
            hovertemplate='<b>Strike:</b> %{x}<br><b>Days:</b> %{y}<br><b>IV:</b> %{z:.2f}%<extra></extra>'
        ))
        
        # Create smooth interpolated surface
        # Create finer grid for smooth surface
        strike_min, strike_max = df['Strike'].min(), df['Strike'].max()
        time_min, time_max = df['TimeToExpiry'].min(), df['TimeToExpiry'].max()
        
        # Create a finer grid (50x50) for smooth interpolation
        strike_fine = np.linspace(strike_min, strike_max, 50)
        time_fine = np.linspace(time_min, time_max, 50)
        strike_grid, time_grid = np.meshgrid(strike_fine, time_fine)
        
        # Prepare data for interpolation
        points = df[['Strike', 'TimeToExpiry']].values
        values = convert_iv_for_display(df[iv_column]).values
        
        # Interpolate IV values onto the fine grid using cubic interpolation
        try:
            iv_grid = griddata(
                points, 
                values, 
                (strike_grid, time_grid), 
                method='cubic',
                fill_value=np.nan
            )
            
            # Fallback to linear if cubic fails
            if np.all(np.isnan(iv_grid)):                
                iv_grid = griddata(points, values, (strike_grid, time_grid), method='linear', fill_value=np.nan)
            
            if not np.all(np.isnan(iv_grid)):
                iv_grid = np.where(iv_grid < 0, np.nan, iv_grid)  # Replace negative values with NaN
                # Also set a reasonable upper bound (e.g., 500% IV is unrealistic)
                iv_grid = np.where(iv_grid > 500, np.nan, iv_grid)
            
            # Add smooth interpolated surface
            if not np.all(np.isnan(iv_grid)):
                fig.add_trace(go.Surface(
                    z=iv_grid,
                    x=strike_grid,
                    y=time_grid,
                    colorscale=PLOT_SETTINGS.get('surface_colorscale', 'Viridis'),
                    opacity=0.5,  # Make surface more transparent to see smile curves
                    showscale=False,
                    name='Smooth Surface',
                    hoverinfo='skip'
                ))
            
            # Volatility smile curves for each expiry as 3D lines
            unique_times = sorted(df['TimeToExpiry'].unique())
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, time_val in enumerate(unique_times):
                expiry_data = df[df['TimeToExpiry'] == time_val].copy()
                if len(expiry_data) >= 2:  # Need at least 2 points for a curve
                    expiry_data = expiry_data.sort_values('Strike')
                    
                    # Get corresponding expiry date for labeling
                    expiry_label = expiry_data['Expiry'].iloc[0] if 'Expiry' in expiry_data.columns else f"{time_val}d"
                    
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter3d(
                        x=expiry_data['Strike'],
                        y=[time_val] * len(expiry_data),  # Constant time for this expiry
                        z=convert_iv_for_display(expiry_data[iv_column]),
                        mode='lines+markers',
                        line=dict(color=color, width=4),
                        marker=dict(size=3, color=color),
                        name=f'Vol Smile - {expiry_label}',
                        hovertemplate=f'<b>Expiry:</b> {expiry_label}<br><b>Strike:</b> %{{x}}<br><b>IV:</b> %{{z:.2f}}%<extra></extra>',
                        showlegend=True
                    ))
        except Exception as e:
            # Fallback to basic grid if interpolation fails
            st.warning(f"Interpolation failed, using basic grid: {str(e)}")
            strike_grid, time_grid = np.meshgrid(strikes, times)
            iv_grid = np.full(strike_grid.shape, np.nan)
            
            for i, time_val in enumerate(times):
                for j, strike_val in enumerate(strikes):
                    subset = df[(df['TimeToExpiry'] == time_val) & (df['Strike'] == strike_val)]
                    if not subset.empty:
                        iv_grid[i, j] = convert_iv_for_display(subset[iv_column].iloc[0])
            
            filled_ratio = np.count_nonzero(~np.isnan(iv_grid)) / iv_grid.size
            if filled_ratio > 0.2:
                fig.add_trace(go.Surface(
                    z=iv_grid,
                    x=strike_grid,
                    y=time_grid,
                    colorscale=PLOT_SETTINGS.get('surface_colorscale', 'Viridis'),
                    opacity=0.6,
                    showscale=False,
                    name='Basic Surface',
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility (%)',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
            ),
            height=PLOT_SETTINGS.get('surface_height', 600),
            template="plotly_dark"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D surface: {str(e)}")
        return None


def create_iv_heatmap_plot(surface_df: pd.DataFrame, iv_column: str = 'VW_IV') -> Optional[go.Figure]:
    """Create IV heatmap with consistent formatting."""
    try:
        if surface_df.empty:
            return None
            
        # Create pivot table
        heatmap_data = surface_df.pivot_table(
            values=iv_column, 
            index='Expiry', 
            columns='Strike', 
            aggfunc='mean'
        )
        
        if heatmap_data.empty:
            return None
        
        # Convert entire DataFrame to display units (percentage); preserves NaN
        heatmap_display = convert_iv_for_display(heatmap_data)
        
        fig = px.imshow(
            heatmap_display,
            labels=dict(x="Strike Price", y="Expiry", color="IV (%)"),
            color_continuous_scale=PLOT_SETTINGS.get('heatmap_colorscale', 'RdYlBu_r'),
            title="Implied Volatility Heatmap"
        )
        
        fig.update_layout(
            height=PLOT_SETTINGS.get('heatmap_height', 500),
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None


def create_volatility_smile(vol_surface: pd.DataFrame, expiry: str, current_price: float) -> Optional[go.Figure]:
    """Create volatility smile plot for specific expiry (market IV only)."""
    try:
        expiry_data = vol_surface[vol_surface['Expiry'] == expiry].copy()
        
        if expiry_data.empty:
            return None
            
        expiry_data = expiry_data.sort_values('Strike')
        
        fig = go.Figure()
        
        # Market IV with consistent display formatting
        fig.add_trace(go.Scatter(
            x=expiry_data['Strike'],
            y=convert_iv_for_display(expiry_data['VW_IV']),
            mode='markers+lines',
            name='Market IV',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Strike:</b> %{x}<br><b>IV:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Add ATM reference line
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
            height=PLOT_SETTINGS.get('smile_height', 400),
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating smile plot: {str(e)}")
        return None


class VolatilitySurfaceProcessor:
    """Handles volatility surface data processing with caching."""
    
    def __init__(self):
        # Remove unused caching fields as they're never used
        pass
    
    def get_processed_surface(self, options_data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], float, Optional[datetime]]:
        """Get processed volatility surface with caching to eliminate repeated calculations."""
        if options_data is None or options_data.empty:
            return None, 0, None
        
        # Use cached calculation with timestamp from data_utils
        from data_utils import get_normalized_vol_surface, get_data_hash
        
        data_hash = get_data_hash(options_data)
        result = get_normalized_vol_surface(options_data, data_hash)
        
        current_price = options_data['Underlying'].iloc[0]
        
        if result and len(result) == 2:
            vol_surface, timestamp = result
            return vol_surface, current_price, timestamp
        else:
            return None, current_price, None
    
    def filter_by_expiry(self, vol_surface: pd.DataFrame, selected_expiry: str) -> pd.DataFrame:
        """Filter surface by expiry."""
        if selected_expiry == 'All Expiries':
            return vol_surface.copy()
        else:
            return vol_surface[vol_surface['Expiry'] == selected_expiry].copy()
    
    def get_summary_metrics(self, vol_surface: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary metrics with consistent IV formatting."""
        if vol_surface.empty:
            return {}
            
        return {
            'avg_iv': convert_iv_for_display(vol_surface['VW_IV'].mean()),
            'max_relative_spread': vol_surface.get('Relative_IV_Spread', pd.Series([0])).max(),
            'total_volume': vol_surface['Total_Volume'].sum(),
            'num_options': len(vol_surface),
            'expiry_range': self._get_expiry_range(vol_surface)
        }
    
    def _get_expiry_range(self, vol_surface: pd.DataFrame) -> Optional[Dict[str, int]]:
        """Get days to expiry range if available."""
        if 'Days_To_Expiry' not in vol_surface.columns:
            return None
            
        return {
            'min_days': int(vol_surface['Days_To_Expiry'].min()),
            'max_days': int(vol_surface['Days_To_Expiry'].max())
        }


class VolatilityTabRenderer:
    """Main renderer for volatility analysis tabs."""
    
    def __init__(self):
        self.processor = VolatilitySurfaceProcessor()
    
    def render_main_tab(self, options_data: pd.DataFrame, current_price: float, symbol: str = "NIFTY"):
        """Render main volatility surface analysis with modular components."""
        if options_data is None or options_data.empty:
            st.error(f"No {symbol} options data available for analysis.")
            return None, None, None
        
        # Process surface data (cached to eliminate repeated calculations)
        vol_surface, current_price, timestamp = self.processor.get_processed_surface(options_data)
        
        # Display data freshness indicator
        if timestamp:
            from data_utils import display_data_freshness
            display_data_freshness(timestamp, symbol)
        
        if vol_surface is None or vol_surface.empty:
            st.error("Unable to calculate volatility surface metrics.")
            return None, None, None
        
        # Render controls
        selected_expiry, max_opportunities = self._render_controls(vol_surface, current_price, symbol)
        
        # Filter data
        vol_surface_filtered = self.processor.filter_by_expiry(vol_surface, selected_expiry)
        
        # Get arbitrage opportunities (use raw options data, not processed surface)
        arbitrage_opportunities = identify_arbitrage_opportunities(options_data, max_opportunities)
        arbitrage_opportunities_filtered = [
            opp for opp in arbitrage_opportunities 
            if selected_expiry == 'All Expiries' or opp.get('Expiry') == selected_expiry
        ]
        
        # Display summary metrics with consistent IV formatting
        self._display_summary_metrics(vol_surface_filtered, len(arbitrage_opportunities_filtered))
        
        # Display visualizations
        self._display_visualizations(vol_surface_filtered, selected_expiry, current_price)
        
        # Display arbitrage analysis (renamed from private to public)
        self.display_mispricing_analysis(arbitrage_opportunities_filtered, symbol, "main_tab")
        
        return options_data, current_price, arbitrage_opportunities
    
    def _render_controls(self, vol_surface: pd.DataFrame, current_price: float, symbol: str) -> Tuple[str, int]:
        """Render UI controls section."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            currency = "₹" if symbol == "NIFTY" else "$"
            st.metric(f"Current {symbol} Price", f"{currency}{current_price:.2f}")
        
        with col2:
            max_opportunities = st.selectbox(
                "Max Arbitrages:",
                options=[10, 20, 30, 50, 100],
                index=1,
                key=f"max_arb_{symbol}_main"
            )
        
        # Expiry filter - sort for stable ordering
        available_expiries = sorted(vol_surface['Expiry'].unique().tolist())
        selected_expiry = st.selectbox(
            "Filter by Expiry:",
            options=['All Expiries'] + available_expiries,
            key=f"expiry_filter_{symbol}_main"
        )
        
        if selected_expiry != 'All Expiries':
            st.info(f"Showing data for expiry: **{selected_expiry}**")
        
        return selected_expiry, max_opportunities
    
    def _display_summary_metrics(self, vol_surface: pd.DataFrame, arbitrage_count: int):
        """Display summary metrics with consistent formatting."""
        metrics = self.processor.get_summary_metrics(vol_surface)
        
        col1, col2, col3 = st.columns(3)  # Remove unused col4
        
        with col1:
            avg_iv = metrics.get('avg_iv', 0)
            st.metric("Avg IV", f"{avg_iv:.1f}%")
        
        with col2:
            st.metric("Arbitrages Found", f"{arbitrage_count}")
        
        with col3:
            total_volume = metrics.get('total_volume', 0)
            st.metric("Total Volume", f"{total_volume:,.0f}")
        
        # Show expiry range 
        expiry_range = metrics.get('expiry_range')
        if expiry_range:
            min_days = expiry_range['min_days']
            max_days = expiry_range['max_days']
            st.info(f"📅 Options expire in {min_days} to {max_days} days")
    
    def _display_visualizations(self, vol_surface: pd.DataFrame, selected_expiry: str, current_price: float):
        """Display volatility surface visualizations."""
        num_expiries = len(vol_surface['Expiry'].unique())
        
        if selected_expiry != 'All Expiries' or num_expiries == 1:
            # Single expiry - show volatility smile
            st.markdown("#### 🎯 Volatility Smile")
            expiry_to_plot = selected_expiry if selected_expiry != 'All Expiries' else vol_surface['Expiry'].iloc[0]
            
            fig_smile = create_volatility_smile(vol_surface, expiry_to_plot, current_price)
            if fig_smile:
                st.plotly_chart(fig_smile, use_container_width=True)
            else:
                st.warning("Unable to create volatility smile plot")
        else:
            # Multiple expiries - show 3D surface and heatmap
            st.markdown("#### 📊 3D Volatility Surface")
            
            fig_3d = create_3d_surface_plot(vol_surface, 'VW_IV', "Market IV 3D Volatility Surface")
            
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Unable to create 3D surface - insufficient data points")
            
            # Heatmap
            st.markdown("#### 🔥 Implied Volatility Heatmap")
            fig_heatmap = create_iv_heatmap_plot(vol_surface)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # IV spread analysis
        self._display_spread_analysis(vol_surface)
    
    def _display_spread_analysis(self, vol_surface: pd.DataFrame):
        """Display IV spread analysis."""
        st.markdown("#### ⚡ Put-Call IV Spread Analysis")
        
        if 'Relative_IV_Spread' not in vol_surface.columns:
            st.info("IV spread analysis not available for this dataset.")
            return
        
        significant_spreads = vol_surface[vol_surface['Relative_IV_Spread'] > 0.05].copy()
        
        if not significant_spreads.empty:
            fig_spread = px.scatter(
                significant_spreads,
                x='Strike',
                y='Relative_IV_Spread',
                size='Total_Volume',
                color='Expiry',
                hover_data=['CE_IV', 'PE_IV', 'Total_Volume'],
                title="Put-Call Relative IV Spreads (>5%)",
                labels={'Relative_IV_Spread': 'Relative IV Spread (%)', 'Strike': 'Strike Price'}
            )
            st.plotly_chart(fig_spread, use_container_width=True)
        else:
            st.info("No significant relative IV spreads (>5%) found.")
    
    def display_mispricing_analysis(self, arbitrage_opportunities: List[Dict], symbol: str, tab_id: str = "main"):
        """Display arbitrage opportunities section (renamed from _display_mispricing_analysis to be public)."""
        st.markdown("#### 🎯 Top Arbitrage Opportunities")
        
        if not arbitrage_opportunities:
            st.info("No arbitrage opportunities detected.")
            return
        
        st.info(f"Showing **{len(arbitrage_opportunities)}** arbitrage opportunities (sorted by profit potential)")
        
        # Separate arbitrage types
        calendar_arbs = [arb for arb in arbitrage_opportunities if arb['Type'] == 'Calendar Arbitrage']
        butterfly_arbs = [arb for arb in arbitrage_opportunities if arb['Type'] == 'Butterfly Arbitrage']
        vertical_arbs = [arb for arb in arbitrage_opportunities if arb['Type'] == 'Vertical Arbitrage']
        
        # Display Calendar Arbitrage
        if calendar_arbs:
            st.markdown("##### 📅 Calendar Arbitrage Opportunities")
            cal_df = pd.DataFrame(calendar_arbs)
            
            # Format display
            if symbol == "NIFTY":
                cal_df['Strike'] = cal_df['Strike'].astype(int)
            else:
                cal_df['Strike'] = cal_df['Strike'].round(2)
            
            for col in ['Profit_Potential', 'Short_Price', 'Long_Price']:
                if col in cal_df.columns:
                    cal_df[col] = cal_df[col].round(2)
            
            calendar_cols = ['Type', 'Strike', 'Short_Expiry', 'Long_Expiry', 'Short_Price', 'Long_Price', 'Profit_Potential', 'Strategy']
            display_cols = [col for col in calendar_cols if col in cal_df.columns]
            st.dataframe(cal_df[display_cols], use_container_width=True)
        
        # Display Butterfly Arbitrage
        if butterfly_arbs:
            st.markdown("##### 🦋 Butterfly Arbitrage Opportunities")
            but_df = pd.DataFrame(butterfly_arbs)
            
            # Format display
            if symbol == "NIFTY":
                for col in ['Strike_Low', 'Strike_Mid', 'Strike_High']:
                    if col in but_df.columns:
                        but_df[col] = but_df[col].astype(int)
            else:
                for col in ['Strike_Low', 'Strike_Mid', 'Strike_High']:
                    if col in but_df.columns:
                        but_df[col] = but_df[col].round(2)
            
            for col in ['Price_Low', 'Price_Mid', 'Price_High', 'Convexity_Violation', 'Profit_Potential', 'Butterfly_Cost']:
                if col in but_df.columns:
                    but_df[col] = but_df[col].round(2)
            
            butterfly_cols = ['Type', 'Strike_Low', 'Strike_Mid', 'Strike_High', 'Expiry', 'Price_Low', 'Price_Mid', 'Price_High', 'Convexity_Violation', 'Profit_Potential', 'Strategy']
            display_cols = [col for col in butterfly_cols if col in but_df.columns]
            st.dataframe(but_df[display_cols], use_container_width=True)
        
        # Display Vertical Arbitrage
        if vertical_arbs:
            st.markdown("##### 📊 Vertical Arbitrage Opportunities")
            vert_df = pd.DataFrame(vertical_arbs)
            
            # Format display
            if symbol == "NIFTY":
                for col in ['Strike_Low', 'Strike_High']:
                    if col in vert_df.columns:
                        vert_df[col] = vert_df[col].astype(int)
            else:
                for col in ['Strike_Low', 'Strike_High']:
                    if col in vert_df.columns:
                        vert_df[col] = vert_df[col].round(2)
            
            for col in ['Price_Low', 'Price_High', 'Profit_Potential']:
                if col in vert_df.columns:
                    vert_df[col] = vert_df[col].round(2)
            
            vertical_cols = ['Type', 'Strike_Low', 'Strike_High', 'Expiry', 'Price_Low', 'Price_High', 'Profit_Potential', 'Strategy']
            display_cols = [col for col in vertical_cols if col in vert_df.columns]
            st.dataframe(vert_df[display_cols], use_container_width=True)
        
        # Combined selection for strategy details
        if len(arbitrage_opportunities) > 0:
            st.markdown("##### Strategy Details")
            
            # Create combined options for selection
            all_arbs_df = pd.DataFrame(arbitrage_opportunities)
            
            selected_idx = st.selectbox(
                "Select arbitrage opportunity for details:",
                options=range(len(all_arbs_df)),
                format_func=lambda x: f"{all_arbs_df.iloc[x]['Type']} - {all_arbs_df.iloc[x]['Strategy']}",
                key=f"arb_details_{symbol}_{tab_id}"
            )
            
            selected_arb = all_arbs_df.iloc[selected_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Arbitrage Type", selected_arb['Type'])
                st.metric("Profit Potential", f"{selected_arb['Profit_Potential']:.2f}")
            
            with col2:
                if 'Expiry' in selected_arb:
                    st.metric("Expiry", selected_arb['Expiry'])
                elif 'Short_Expiry' in selected_arb:
                    st.metric("Short Expiry", selected_arb['Short_Expiry'])
                    st.metric("Long Expiry", selected_arb['Long_Expiry'])
                
                st.write("**Strategy:**")
                st.write(selected_arb['Strategy'])
            
            # Show detailed metrics based on arbitrage type
            if selected_arb['Type'] == 'Calendar Arbitrage':
                currency = "₹" if symbol == "NIFTY" else "$"
                st.info(f"📅 Calendar Spread: Short {selected_arb['Short_Expiry']} Call @ {currency}{selected_arb['Short_Price']:.2f}, Long {selected_arb['Long_Expiry']} Call @ {currency}{selected_arb['Long_Price']:.2f}")
            elif selected_arb['Type'] == 'Butterfly Arbitrage':
                currency = "₹" if symbol == "NIFTY" else "$"
                st.info(f"🦋 Butterfly: Buy {selected_arb['Strike_Low']} @ {currency}{selected_arb['Price_Low']:.2f}, Sell 2x {selected_arb['Strike_Mid']} @ {currency}{selected_arb['Price_Mid']:.2f}, Buy {selected_arb['Strike_High']} @ {currency}{selected_arb['Price_High']:.2f}")
                st.warning(f"⚠️ Convexity Violation: {selected_arb['Convexity_Violation']:.3f}")
            elif selected_arb['Type'] == 'Vertical Arbitrage':
                currency = "₹" if symbol == "NIFTY" else "$"
                st.info(f"📊 Vertical Spread: Buy {selected_arb['Strike_Low']} @ {currency}{selected_arb['Price_Low']:.2f}, Sell {selected_arb['Strike_High']} @ {currency}{selected_arb['Price_High']:.2f}")


def render_volatility_tabs(sidebar_params: Dict[str, Any]):
    """
    Main entry point for volatility analysis tabs.
    
    Args:
        sidebar_params: Dictionary containing sidebar parameters including symbol selection
    """
    # Get symbol from sidebar params
    symbol = sidebar_params.get('symbol', 'NIFTY')
    
    # Create subtabs for different analyses
    vol_tab1, vol_tab2 = st.tabs([
        f"📊 {symbol} Volatility Surface",
        "🎯 Arbitrage Scanner"
    ])
    
    # Initialize renderer
    renderer = VolatilityTabRenderer()
    
    with vol_tab1:
        st.markdown(f"### 📊 {symbol} Volatility Surface Analysis")
        
        # Fetch options data based on symbol
        if symbol == "NIFTY":
            raw_data = fetch_nifty_options()
            options_data = process_nifty_options_data(raw_data)
            current_price = sidebar_params.get('current_price', 25000)  # Default NIFTY price
        else:
            # For US stocks, fetch data using yfinance
            from nifty_data import fetch_us_options, process_us_options_data
            raw_data = fetch_us_options(symbol)
            if raw_data:
                options_data = process_us_options_data(raw_data)
                current_price = raw_data.get('current_price', sidebar_params.get('current_price', 100))
            else:
                st.error(f"Unable to fetch options data for {symbol}. Please check the symbol and try again.")
                options_data = None
                current_price = sidebar_params.get('current_price', 100)
        
        if options_data is not None and (isinstance(options_data, pd.DataFrame) and not options_data.empty):
            # Process and display the volatility surface
            processed_data, current_price_actual, arbitrage_opps = renderer.render_main_tab(
                options_data, current_price, symbol
            )
        else:
            st.error(f"No options data available for {symbol}")
    
    # with vol_tab2:
    #     st.markdown("### 🎯 Advanced Arbitrage Scanner")
        
    #     if symbol == "NIFTY":
    #         raw_data = fetch_nifty_options()
    #         options_data = process_nifty_options_data(raw_data)
    #     else:
    #         # For US stocks, fetch data using yfinance
    #         from nifty_data import fetch_us_options, process_us_options_data
    #         raw_data = fetch_us_options(symbol)
    #         if raw_data:
    #             options_data = process_us_options_data(raw_data)
    #         else:
    #             st.error(f"Unable to fetch options data for {symbol}")
    #             options_data = None
                
    #     if options_data is not None and (isinstance(options_data, pd.DataFrame) and not options_data.empty):
    #         # Focus on arbitrage detection
    #         arbitrage_opportunities = identify_arbitrage_opportunities(options_data, max_opportunities=50)
            
    #         if arbitrage_opportunities:
    #             st.success(f"Found {len(arbitrage_opportunities)} potential arbitrage opportunities!")
                
    #             # Create detailed arbitrage analysis (using public method)
    #             renderer.display_mispricing_analysis(arbitrage_opportunities, symbol, "scanner_tab")
                
    #             # Add summary statistics
    #             st.markdown("#### 📈 Arbitrage Summary")
                
    #             # Count by type
    #             arb_types = {}
    #             total_profit = 0
    #             for arb in arbitrage_opportunities:
    #                 arb_type = arb['Type']
    #                 arb_types[arb_type] = arb_types.get(arb_type, 0) + 1
    #                 total_profit += arb.get('Profit_Potential', 0)
                
    #             col1, col2, col3, col4 = st.columns(4)
    #             with col1:
    #                 st.metric("Total Opportunities", len(arbitrage_opportunities))
    #             with col2:
    #                 st.metric("Calendar Spreads", arb_types.get('Calendar Arbitrage', 0))
    #             with col3:
    #                 st.metric("Butterfly Spreads", arb_types.get('Butterfly Arbitrage', 0))
    #             with col4:
    #                 st.metric("Vertical Spreads", arb_types.get('Vertical Arbitrage', 0))
                
    #             currency_symbol = "₹" if symbol == "NIFTY" else "$"
    #             st.info(f"💰 Total potential profit across all opportunities: {currency_symbol}{total_profit:.2f}")
    #         else:
    #             st.info("No arbitrage opportunities detected in current market conditions.")
    #     else:
    #         st.error(f"Unable to fetch options data for arbitrage analysis for {symbol}.")
