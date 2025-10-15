import pandas as pd
from datetime import datetime
from nsepython import nse_optionchain_scrapper
import numpy as np
import streamlit as st
import yfinance as yf
from typing import Optional, List, Dict, Any

@st.cache_data(ttl=60)  # Cache for 60 seconds to prevent stale data
def fetch_nifty_options():
    """Fetch NIFTY options data using nsepython package."""
    try:
        data = nse_optionchain_scrapper('NIFTY')
        return data
    except Exception as e:
        st.error(f"Error fetching NIFTY data: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Add TTL here too since it processes live data
def process_nifty_options_data(data):
    """Process the raw NIFTY options data into a structured DataFrame supporting multiple expiries."""
    if not data:
        return None
    records = []
    try:
        if 'records' in data and 'data' in data['records']:
            expiry_dates = data['records'].get('expiryDates', ['Unknown'])
            for item in data['records']['data']:
                strike_price = item.get('strikePrice')
                ce_data = item.get('CE')
                pe_data = item.get('PE')
                # Use expiry from CE/PE if present, else fallback
                if ce_data:
                    ce_expiry = ce_data.get('expiryDate') or expiry_dates[0]
                    ce_bid = ce_data.get('bidprice', 0) or 0
                    ce_ask = ce_data.get('askPrice', 0) or 0
                    ce_mid_price = (ce_bid + ce_ask) / 2 if (ce_bid > 0 and ce_ask > 0) else ce_data.get('lastPrice', 0)
                    
                    records.append({
                        'Symbol': 'NIFTY',
                        'Type': 'CE',
                        'Strike': strike_price,
                        'Expiry': ce_expiry,
                        'Open_Interest': ce_data.get('openInterest'),
                        'Change_in_OI': ce_data.get('changeinOpenInterest'),
                        'Volume': ce_data.get('totalTradedVolume'),
                        'IV': ce_data.get('impliedVolatility'),
                        'Last_Price': ce_data.get('lastPrice'),
                        'Underlying': ce_data.get('underlyingValue'),
                        'Bid': ce_bid,
                        'Ask': ce_ask,
                        'Mid_Price': ce_mid_price,
                        'Timestamp': datetime.now()
                    })
                if pe_data:
                    pe_expiry = pe_data.get('expiryDate') or expiry_dates[0]
                    pe_bid = pe_data.get('bidprice', 0) or 0
                    pe_ask = pe_data.get('askPrice', 0) or 0
                    pe_mid_price = (pe_bid + pe_ask) / 2 if (pe_bid > 0 and pe_ask > 0) else pe_data.get('lastPrice', 0)
                    
                    records.append({
                        'Symbol': 'NIFTY',
                        'Type': 'PE',
                        'Strike': strike_price,
                        'Expiry': pe_expiry,
                        'Open_Interest': pe_data.get('openInterest'),
                        'Change_in_OI': pe_data.get('changeinOpenInterest'),
                        'Volume': pe_data.get('totalTradedVolume'),
                        'IV': pe_data.get('impliedVolatility'),
                        'Last_Price': pe_data.get('lastPrice'),
                        'Underlying': pe_data.get('underlyingValue'),
                        'Bid': pe_bid,
                        'Ask': pe_ask,
                        'Mid_Price': pe_mid_price,
                        'Timestamp': datetime.now()
                    })
        df = pd.DataFrame(records)
        df = df.dropna(subset=['Underlying'])

        if not df.empty:
            df.sort_values(by=['Expiry', 'Type', 'Strike'], inplace=True)
            if st.session_state.get('debug_mode', False):  # Only show if debug mode enabled
                st.write('DEBUG: Unique expiries found:', df['Expiry'].unique())
        return df
    except Exception as e:
        st.error(f"Error processing NIFTY data: {str(e)}")
        return None

def calculate_vol_surface_metrics(df):
    """Calculate volatility surface metrics and identify potential mispricings."""
    if df is None or df.empty:
        return None
    
    # Group by expiry and strike to ensure proper CE/PE pairing
    surface_data = []
    
    # Correct loop: process each (expiry, strike) pair
    for expiry in df['Expiry'].unique():
        expiry_data = df[df['Expiry'] == expiry]
        for strike in expiry_data['Strike'].unique():
            strike_data = expiry_data[expiry_data['Strike'] == strike]
            ce_data = strike_data[strike_data['Type'] == 'CE']
            pe_data = strike_data[strike_data['Type'] == 'PE']
            
            if not ce_data.empty and not pe_data.empty:
                # Safely extract values with NaN handling
                ce_iv_raw = ce_data['IV'].iloc[0]
                pe_iv_raw = pe_data['IV'].iloc[0]
                ce_iv = float(ce_iv_raw) if pd.notna(ce_iv_raw) else 0.0
                pe_iv = float(pe_iv_raw) if pd.notna(pe_iv_raw) else 0.0
                
                # Calculate put-call IV spread (potential mispricing indicator)
                mean_iv = (ce_iv + pe_iv) / 2.0
                iv_spread = abs(ce_iv - pe_iv)
                relative_iv_spread = iv_spread / max(mean_iv, 1.0) if mean_iv > 0 else 0.0
                
                # Volume fields
                ce_vol_raw = ce_data['Volume'].iloc[0]
                pe_vol_raw = pe_data['Volume'].iloc[0]
                ce_volume = int(ce_vol_raw) if pd.notna(ce_vol_raw) else 0
                pe_volume = int(pe_vol_raw) if pd.notna(pe_vol_raw) else 0
                total_volume = ce_volume + pe_volume
                
                # Volume-weighted IV
                if total_volume > 0:
                    vw_iv = (ce_iv * ce_volume + pe_iv * pe_volume) / total_volume
                else:
                    vw_iv = mean_iv
                
                surface_data.append({
                    'Strike': strike,
                    'Expiry': expiry,
                    'CE_IV': ce_iv,
                    'PE_IV': pe_iv,
                    'IV_Spread': iv_spread,
                    'Relative_IV_Spread': relative_iv_spread,
                    'VW_IV': vw_iv,
                    'CE_Volume': ce_volume,
                    'PE_Volume': pe_volume,
                    'Total_Volume': total_volume,
                    'CE_OI': ce_data['Open_Interest'].iloc[0] if pd.notna(ce_data['Open_Interest'].iloc[0]) else 0,
                    'PE_OI': pe_data['Open_Interest'].iloc[0] if pd.notna(pe_data['Open_Interest'].iloc[0]) else 0,
                    'CE_Price': ce_data['Last_Price'].iloc[0] if pd.notna(ce_data['Last_Price'].iloc[0]) else 0,
                    'PE_Price': pe_data['Last_Price'].iloc[0] if pd.notna(pe_data['Last_Price'].iloc[0]) else 0,
                    'Underlying': ce_data['Underlying'].iloc[0] if not ce_data.empty else pe_data['Underlying'].iloc[0]
                })
    
    surface_df = pd.DataFrame(surface_data)
    
    if not surface_df.empty:
        # Calculate additional mispricing indicators
        surface_df['Moneyness'] = surface_df['Strike'] / surface_df['Underlying']
        surface_df['ATM_Distance'] = abs(surface_df['Strike'] - surface_df['Underlying'])
        
        # Calculate time to expiry in days and years
        try:
            surface_df['Expiry_Date'] = pd.to_datetime(surface_df['Expiry'], format='%d-%b-%Y', errors='coerce')
            surface_df['Expiry_Date'] = surface_df['Expiry_Date'].fillna(
                pd.to_datetime(surface_df['Expiry'], format='%Y-%m-%d', errors='coerce')
            )
            today = pd.Timestamp.now().normalize()
            surface_df['Days_To_Expiry'] = (surface_df['Expiry_Date'] - today).dt.days
            surface_df['Time_To_Expiry'] = surface_df['Days_To_Expiry'] / 365.0
            surface_df = surface_df[surface_df['Days_To_Expiry'] >= 0]
        except Exception:
            surface_df['Days_To_Expiry'] = 30
            surface_df['Time_To_Expiry'] = 30 / 365.0
        
        if surface_df.empty:
            return None
        
        # Mispricing criteria
        relative_spread_threshold = surface_df['Relative_IV_Spread'].quantile(0.9)
        min_volume_threshold = surface_df['Total_Volume'].quantile(0.25)
        surface_df['Potential_Mispricing'] = (
            (surface_df['Relative_IV_Spread'] > relative_spread_threshold) &
            (surface_df['Total_Volume'] > min_volume_threshold) &
            (surface_df['Relative_IV_Spread'] > 0.05)
        )
        surface_df['IV_Rank'] = surface_df['VW_IV'].rank(pct=True) * 100
    
    return surface_df

def create_3d_vol_surface_data(df):
    """Create 3D volatility surface data for visualization with axes: strike/moneyness, IV, expiry."""
    if df is None or df.empty:
        return None, None, None, None, None
    try:
        surface_df = calculate_vol_surface_metrics(df)
        if surface_df is None or surface_df.empty:
            return None, None, None, None, None
        # Get unique strikes, moneyness, and expiries
        strikes = sorted(surface_df['Strike'].unique())
        moneyness = sorted(surface_df['Moneyness'].unique())
        expiries = sorted(surface_df['Expiry'].unique())
        # Create meshgrid for strike/moneyness and expiry
        n_strikes = min(len(strikes), 20)
        n_moneyness = min(len(moneyness), 15)
        n_expiry = len(expiries)
        strike_range = np.linspace(min(strikes), max(strikes), n_strikes)
        moneyness_range = np.linspace(min(moneyness), max(moneyness), n_moneyness)
        expiry_range = expiries
        # Prepare 3D grid for IV values
        iv_grid = np.zeros((n_moneyness, n_strikes, n_expiry))
        # Fill IV grid for each expiry
        for k, expiry in enumerate(expiry_range):
            expiry_df = surface_df[surface_df['Expiry'] == expiry]
            for i, moneyness_val in enumerate(moneyness_range):
                for j, strike_val in enumerate(strike_range):
                    # Find closest data point for this expiry
                    distances = np.abs(expiry_df['Strike'] - strike_val) + 100 * np.abs(expiry_df['Moneyness'] - moneyness_val)
                    if not distances.empty:
                        closest_idx = distances.idxmin()
                        if distances[closest_idx] < 200:
                            iv_grid[i, j, k] = expiry_df.loc[closest_idx, 'VW_IV']
                        else:
                            nearby_points = expiry_df[distances < 500]
                            if not nearby_points.empty:
                                weights = 1 / (distances[nearby_points.index] + 1)
                                iv_grid[i, j, k] = np.average(nearby_points['VW_IV'], weights=weights)
                            else:
                                iv_grid[i, j, k] = expiry_df['VW_IV'].mean() if not expiry_df.empty else 0
                    else:
                        iv_grid[i, j, k] = 0
        # Return grids for visualization
        return strike_range, moneyness_range, expiry_range, iv_grid, surface_df
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.write(f"Error in create_3d_vol_surface_data: {e}")
        return None, None, None, None, None

def create_strike_expiry_iv_surface(surface_df):
    """Create a 2D grid of IV values for each (strike, expiry) pair, averaging across moneyness."""
    if surface_df is None or surface_df.empty:
        return None, None, None
    strikes = sorted(surface_df['Strike'].unique())
    expiries = sorted(surface_df['Expiry'].unique())
    n_strikes = len(strikes)
    n_expiry = len(expiries)
    iv_surface = np.zeros((n_strikes, n_expiry))
    for i, strike in enumerate(strikes):
        for j, expiry in enumerate(expiries):
            subset = surface_df[(surface_df['Strike'] == strike) & (surface_df['Expiry'] == expiry)]
            if not subset.empty:
                iv_surface[i, j] = subset['VW_IV'].mean()
            else:
                iv_surface[i, j] = np.nan  # Use NaN for missing data
    return strikes, expiries, iv_surface

@st.cache_data(ttl=60)  # Add TTL for US options too
def fetch_us_options(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch US options data using yfinance package.
    
    Args:
        symbol: US stock symbol (e.g., 'AAPL', 'SPY', 'TSLA')
    
    Returns:
        Dictionary containing options data or None if error
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current stock price
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not current_price:
            # Try to get from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                st.error(f"Could not fetch current price for {symbol}")
                return None
        
        # Get available expiration dates
        expiry_dates = ticker.options
        
        if not expiry_dates:
            st.error(f"No options data available for {symbol}")
            return None
        
        options_data = {
            'symbol': symbol.upper(),
            'current_price': current_price,
            'expiry_dates': list(expiry_dates),
            'chains': {}
        }
        
        # Fetch option chain for each expiry (limit to first 6 for performance)
        for expiry in expiry_dates[:6]:
            try:
                chain = ticker.option_chain(expiry)
                options_data['chains'][expiry] = {
                    'calls': chain.calls,
                    'puts': chain.puts
                }
            except Exception as e:
                st.warning(f"Could not fetch options for {symbol} expiry {expiry}: {str(e)}")
                continue
        
        return options_data
        
    except Exception as e:
        st.error(f"Error fetching US options data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Add TTL for processing US data too
def process_us_options_data(options_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Process US options data from yfinance into a structured DataFrame.
    
    Args:
        options_data: Dictionary containing options data from fetch_us_options
    
    Returns:
        DataFrame with processed options data
    """
    if not options_data or 'chains' not in options_data:
        return None
    
    records = []
    symbol = options_data['symbol']
    current_price = options_data['current_price']
    
    try:
        for expiry, chain_data in options_data['chains'].items():
            # Process calls
            if 'calls' in chain_data and not chain_data['calls'].empty:
                calls_df = chain_data['calls']
                for _, row in calls_df.iterrows():
                    bid = row.get('bid', 0) or 0
                    ask = row.get('ask', 0) or 0
                    mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else row.get('lastPrice', 0)
                    
                    records.append({
                        'Symbol': symbol,
                        'Type': 'CE',  # Call European style notation for consistency
                        'Strike': row.get('strike'),
                        'Expiry': expiry,
                        'Open_Interest': row.get('openInterest', 0),
                        'Change_in_OI': None,  # Not available in yfinance
                        'Volume': row.get('volume', 0),
                        'IV': row.get('impliedVolatility', 0) * 100 if pd.notna(row.get('impliedVolatility')) else 0,  # Convert to percentage
                        'Last_Price': row.get('lastPrice', 0),
                        'Underlying': current_price,
                        'Bid': bid,
                        'Ask': ask,
                        'Mid_Price': mid_price,
                        'Contract_Symbol': row.get('contractSymbol', ''),
                        'Timestamp': datetime.now()
                    })
            
            # Process puts
            if 'puts' in chain_data and not chain_data['puts'].empty:
                puts_df = chain_data['puts']
                for _, row in puts_df.iterrows():
                    bid = row.get('bid', 0) or 0
                    ask = row.get('ask', 0) or 0
                    mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else row.get('lastPrice', 0)
                    
                    records.append({
                        'Symbol': symbol,
                        'Type': 'PE',  # Put European style notation for consistency
                        'Strike': row.get('strike'),
                        'Expiry': expiry,
                        'Open_Interest': row.get('openInterest', 0),
                        'Change_in_OI': None,  # Not available in yfinance
                        'Volume': row.get('volume', 0),
                        'IV': row.get('impliedVolatility', 0) * 100 if pd.notna(row.get('impliedVolatility')) else 0,  # Convert to percentage
                        'Last_Price': row.get('lastPrice', 0),
                        'Underlying': current_price,
                        'Bid': bid,
                        'Ask': ask,
                        'Mid_Price': mid_price,
                        'Contract_Symbol': row.get('contractSymbol', ''),
                        'Timestamp': datetime.now()
                    })
        
        df = pd.DataFrame(records)
        df = df.dropna(subset=['Underlying'])

        if not df.empty:
            # Filter out options with zero IV or last price (likely inactive)
            df = df[(df['IV'] > 0) & (df['Last_Price'] > 0)]
            df.sort_values(by=['Expiry', 'Type', 'Strike'], inplace=True)
            
            # Add moneyness calculation
            df['Moneyness'] = df['Strike'] / df['Underlying']
            
            if st.session_state.get('debug_mode', False):  # Only show if debug mode enabled
                st.write(f'DEBUG: Fetched {len(df)} US options for {symbol}')
                st.write('DEBUG: Unique expiries found:', df['Expiry'].unique())
        
        return df
        
    except Exception as e:
        st.error(f"Error processing US options data: {str(e)}")
        return None

def get_popular_us_symbols() -> List[str]:
    """Get list of popular US symbols for options trading."""
    return [
        'SPY',    # S&P 500 ETF
        'QQQ',    # NASDAQ ETF
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'TSLA',   # Tesla
        'NVDA',   # NVIDIA
        'META',   # Meta
        'NFLX',   # Netflix
        'DIS',    # Disney
        'BA',     # Boeing
        'GE',     # General Electric
        'F',      # Ford
        'AMD',    # AMD
        'INTC',   # Intel
        'JPM',    # JPMorgan Chase
        'BAC',    # Bank of America
        'XOM',    # Exxon Mobil
        'JNJ'     # Johnson & Johnson
    ]

@st.cache_data
def validate_us_symbol(symbol: str) -> bool:
    """
    Validate if a US symbol has options data available.
    
    Args:
        symbol: US stock symbol to validate
    
    Returns:
        True if symbol has options, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expiry_dates = ticker.options
        return len(expiry_dates) > 0
    except (ValueError, KeyError, ConnectionError) as e:
        if st.session_state.get('debug_mode', False):
            st.write(f"Symbol validation error for {symbol}: {str(e)}")
        return False
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.write(f"Unexpected error validating {symbol}: {str(e)}")
        return False
