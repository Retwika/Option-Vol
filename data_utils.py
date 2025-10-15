"""
Data utilities for volatility surface analysis.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional, List
from datetime import datetime
from config import PLOT_SETTINGS


def normalize_iv(df: pd.DataFrame, iv_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standardize IV units - convert percentage to decimal if needed.
    
    Args:
        df: DataFrame with IV columns
        iv_columns: List of IV column names to normalize. If None, auto-detects.
    
    Returns:
        DataFrame with normalized IV columns (decimal format)
    """
    if df is None or df.empty:
        return df
    
    df_normalized = df.copy()
    
    # Auto-detect IV columns if not specified
    if iv_columns is None:
        iv_columns = [col for col in df.columns if 'IV' in col and col != 'IV_Spread']
    
    for col in iv_columns:
        if col in df_normalized.columns:
            max_val = df_normalized[col].max()
            if pd.notna(max_val) and max_val > 3.0:  # Likely in percentage form
                df_normalized[col] = df_normalized[col] / 100.0
    
    return df_normalized


def convert_iv_for_display(value):
    """Convert decimal IV to percentage for display (robust for scalar, Series, DataFrame, ndarray)."""
    # Vectorized containers: just multiply; NaNs are preserved
    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray)):
        return value * 100
    
    # Scalar handling: preserve NaN
    try:
        if pd.isna(value):
            return value
    except Exception:
        pass
    
    return value * 100


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_normalized_vol_surface(options_data, data_hash: Optional[str] = None):
    """
    Get normalized volatility surface with caching.
    
    Args:
        options_data: Raw options DataFrame
        data_hash: Hash for cache key (optional)
    
    Returns:
        Tuple of (normalized volatility surface DataFrame, timestamp)
    """
    from nifty_data import calculate_vol_surface_metrics
    
    if options_data is None or options_data.empty:
        return None, None
    
    # Calculate surface metrics
    vol_surface = calculate_vol_surface_metrics(options_data)
    
    if vol_surface is None or vol_surface.empty:
        return None, None
    
    # Normalize IV columns
    iv_columns = ['VW_IV', 'CE_IV', 'PE_IV']
    vol_surface_normalized = normalize_iv(vol_surface, iv_columns)
    
    # Add timestamp for when data was processed
    timestamp = datetime.now()
    
    return vol_surface_normalized, timestamp


def get_data_hash(df: pd.DataFrame) -> str:
    """Generate a simple hash for DataFrame caching (numeric-safe)."""
    if df is None or df.empty:
        return "empty"
    
    try:
        df_num = df.select_dtypes(include=[np.number])
        if df_num is not None and not df_num.empty:
            first_sum = float(df_num.iloc[0].sum()) if len(df_num) > 0 else 0.0
            last_sum = float(df_num.iloc[-1].sum()) if len(df_num) > 0 else 0.0
            hash_components = [
                str(df.shape),
                f"{first_sum:.6f}",
                f"{last_sum:.6f}"
            ]
        else:
            # Fallback: use a couple of string identifiers if no numeric columns
            first_key = str(df.iloc[0].get('Expiry', 'first')) if len(df) > 0 else 'first'
            last_key = str(df.iloc[-1].get('Expiry', 'last')) if len(df) > 0 else 'last'
            hash_components = [str(df.shape), first_key, last_key]
        return "_".join(hash_components)
    except Exception:
        # Final fallback to shape-only hash
        return str(df.shape)


def display_data_freshness(timestamp: datetime, symbol: str):
    """
    Display data freshness indicator with manual refresh option.
    
    Args:
        timestamp: When data was last fetched
        symbol: Symbol being analyzed
    """
    if timestamp is None:
        return
    
    # Calculate time since last update
    now = datetime.now()
    time_diff = now - timestamp
    minutes_ago = int(time_diff.total_seconds() / 60)
    
    # Create columns for timestamp and refresh button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if minutes_ago < 1:
            st.success(f"📊 {symbol} data: Fresh (just updated)")
        elif minutes_ago < 5:
            st.info(f"📊 {symbol} data: Updated {minutes_ago} minute(s) ago")
        else:
            st.warning(f"📊 {symbol} data: Updated {minutes_ago} minute(s) ago (auto-refresh in {5-minutes_ago} min)")
    
    with col2:
        if st.button("🔄 Refresh", key=f"refresh_{symbol}", help="Manually refresh data"):
            # Clear the cache for this function
            get_normalized_vol_surface.clear()
            st.rerun()
