"""
UI components for the Streamlit app.
"""

import streamlit as st
import numpy as np
from nifty_data import (
    fetch_nifty_options, process_nifty_options_data,
    fetch_us_options, process_us_options_data, 
    get_popular_us_symbols, validate_us_symbol
)

def render_sidebar(strategy):
    """Render the sidebar with global parameters."""
    with st.sidebar:
        st.header("🎛️ Global Parameters")
        
        # Add debug mode toggle
        debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Show debug information and logs")
        st.session_state['debug_mode'] = debug_mode
        
        # Data source selector
        data_source = st.radio(
            "Data Source",
            ["Manual Input", "NIFTY Live Data", "US Options (Yahoo Finance)"],
            help="Choose where to get option data from"
        )
        
        # Initialize variables
        nifty_df = None
        us_df = None
        current_price = 100.0  # Default value
        selected_symbol = None  # Initialize symbol variable
        
        if data_source == "NIFTY Live Data":
            with st.spinner("Fetching live NIFTY options data..."):
                try:
                    raw_data = fetch_nifty_options()
                    nifty_df = process_nifty_options_data(raw_data)
                    if nifty_df is not None and not nifty_df.empty:
                        st.success("✅ Live NIFTY data loaded successfully!")
                        current_price = float(nifty_df['Underlying'].iloc[0])
                        st.metric("NIFTY Spot Price", f"₹{current_price:.2f}")
                    else:
                        st.error("❌ Failed to load NIFTY data. Falling back to manual input.")
                        data_source = "Manual Input"
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.warning("Falling back to manual input.")
                    data_source = "Manual Input"
        
        elif data_source == "US Options (Yahoo Finance)":
            st.subheader("🇺🇸 US Symbol Selection")
            
            # Symbol input method
            symbol_method = st.radio(
                "Symbol Input Method",
                ["Popular Symbols", "Custom Symbol"],
                horizontal=True
            )
            
            if symbol_method == "Popular Symbols":
                popular_symbols = get_popular_us_symbols()
                selected_symbol = st.selectbox(
                    "Select Popular Symbol",
                    popular_symbols,
                    help="Choose from popular US options symbols"
                )
            else:
                selected_symbol = st.text_input(
                    "Enter Symbol",
                    value="AAPL",
                    help="Enter US stock symbol (e.g., AAPL, MSFT, SPY)"
                ).upper()
            
            if selected_symbol:
                # Validate symbol
                if validate_us_symbol(selected_symbol):
                    with st.spinner(f"Fetching {selected_symbol} options data..."):
                        try:
                            us_options_data = fetch_us_options(selected_symbol)
                            if us_options_data:
                                us_df = process_us_options_data(us_options_data)
                                if us_df is not None and not us_df.empty:
                                    st.success(f"✅ {selected_symbol} options data loaded!")
                                    current_price = float(us_df['Underlying'].iloc[0])
                                    st.metric(f"{selected_symbol} Price", f"${current_price:.2f}")
                                    
                                    # Show expiry dates available
                                    expiries = us_df['Expiry'].unique()
                                    st.info(f"📅 {len(expiries)} expiry dates available")
                                else:
                                    st.error(f"❌ Failed to process {selected_symbol} data.")
                                    data_source = "Manual Input"
                            else:
                                st.error(f"❌ Failed to fetch {selected_symbol} data.")
                                data_source = "Manual Input"
                        except Exception as e:
                            st.error(f"❌ Error fetching {selected_symbol}: {str(e)}")
                            data_source = "Manual Input"
                else:
                    st.error(f"❌ Symbol '{selected_symbol}' not found or has no options data.")
                    data_source = "Manual Input"
        
        # Market parameters
        st.subheader("Market Conditions")
        if data_source == "Manual Input":
            current_price = st.number_input(
                "Current Underlying Price", 
                value=100.0, 
                min_value=0.01,
                help="Current price of the underlying asset"
            )
        
        # Show IV selector based on data source
        if data_source == "NIFTY Live Data" and nifty_df is not None:
            avg_iv = nifty_df['IV'].mean() / 100
            iv = avg_iv
            st.metric("Average Implied Volatility", f"{avg_iv:.2%}")
        elif data_source == "US Options (Yahoo Finance)" and us_df is not None:
            avg_iv = us_df['IV'].mean() / 100
            iv = avg_iv
            st.metric("Average Implied Volatility", f"{avg_iv:.2%}")
        else:
            iv = st.slider(
                "Implied Volatility (σ)", 
                min_value=0.05, 
                max_value=1.0, 
                value=0.2, 
                step=0.01,
                help="Expected volatility of the underlying asset"
            )
        
        r = st.number_input(
            "Risk-Free Rate (r)", 
            value=0.01, 
            min_value=0.0,
            max_value=0.20,
            step=0.001,
            format="%.3f",
            help="Annual risk-free interest rate"
        )
        
        days_to_expiry = st.slider(
            "Days to Expiry", 
            min_value=1, 
            max_value=365, 
            value=30,
            help="Number of days until option expiry"
        )
        
        T = days_to_expiry / 365
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.info("💡 **Tip:** Adjust these parameters to see their effect on payoffs and Greeks.")
        
        # Strategy management
        st.markdown("### 🔧 Strategy Management")
        if st.button("🔄 Reset All Legs", use_container_width=True):
            strategy.clear_legs()
            st.rerun()
    
    # Determine the symbol based on data source
    if data_source == "US Options (Yahoo Finance)":
        symbol = selected_symbol
    elif data_source == "NIFTY Live Data":
        symbol = "NIFTY"
    else:
        symbol = "Manual"  # For manual input mode
    
    return {
        'data_source': data_source,
        'nifty_df': nifty_df,
        'us_df': us_df,
        'current_price': current_price,
        'iv': iv,
        'r': r,
        'T': T,
        'days_to_expiry': days_to_expiry,
        'symbol': symbol
    }

def render_strategy_input_form(strategy, current_price, currency_symbol="₹"):
    """Render the option leg input form."""
    st.markdown("### 📝 Build Your Strategy")
    with st.expander("➕ Add Option Leg", expanded=not bool(strategy)):
        with st.form("option_leg_form", clear_on_submit=True):
            input_cols = st.columns(6)
            
            option_type = input_cols[0].selectbox(
                "Type", 
                ["Call", "Put"],
                help="Choose Call or Put option"
            )
            
            action = input_cols[1].selectbox(
                "Action", 
                ["Buy", "Sell"],
                help="Buy or Sell the option"
            )
            
            strike_price = input_cols[2].number_input(
                f"Strike ({currency_symbol})", 
                value=current_price,
                min_value=0.01,
                help="Strike price of the option"
            )
            
            premium = input_cols[3].number_input(
                f"Premium ({currency_symbol})", 
                value=5.0,
                min_value=0.01,
                help="Option premium per contract"
            )
            
            quantity = input_cols[4].number_input(
                "Contracts", 
                min_value=1, 
                value=1,
                help="Number of contracts"
            )
            
            add_leg = input_cols[5].form_submit_button(
                "➕ Add Leg",
                use_container_width=True
            )
            
            if add_leg:
                strategy.add_leg(option_type, action, strike_price, premium, quantity)
                st.success(f"✅ Added: {action} {quantity} {option_type} @ {currency_symbol}{strike_price}")
                st.rerun()

def render_strategy_templates(strategy, current_price, currency_symbol="₹"):
    """Render quick strategy templates."""
    st.markdown("### 🎯 Quick Strategy Templates")
    template_cols = st.columns(3)
    
    with template_cols[0]:
        if st.button("📈 Long Call", use_container_width=True):
            strategy.clear_legs()
            strategy.add_leg("Call", "Buy", current_price, 5.0, 1)
            st.rerun()
    
    with template_cols[1]:
        if st.button("📉 Long Put", use_container_width=True):
            strategy.clear_legs()
            strategy.add_leg("Put", "Buy", current_price, 5.0, 1)
            st.rerun()
    
    with template_cols[2]:
        if st.button("🔄 Long Straddle", use_container_width=True):
            strategy.clear_legs()
            strategy.add_leg("Call", "Buy", current_price, 5.0, 1)
            strategy.add_leg("Put", "Buy", current_price, 5.0, 1)
            st.rerun()

def render_welcome_message():
    """Render welcome message when no strategy is built."""
    st.markdown("### 👋 Welcome to the Option Strategy Visualizer!")
    st.info("""
    🚀 **Get Started:**
    1. Set your market parameters in the sidebar
    2. Add option legs using the form above
    3. Analyze your strategy with interactive charts and Greeks

          """)

def render_strategy_summary(strategy):
    """Render strategy summary with leg details."""
    st.markdown("### 🧩 Current Strategy")
    
    with st.expander("📋 Strategy Details", expanded=True):
        legs_summary = strategy.get_strategy_summary()
        
        for i, leg in enumerate(legs_summary):
            col_leg, col_remove = st.columns([4, 1])
            
            with col_leg:
                st.markdown(f"**Leg {i+1}:** {leg['description']}")
            
            with col_remove:
                if st.button(f"❌", key=f"remove_{i}", help="Remove this leg"):
                    strategy.remove_leg(i)
                    st.rerun()
