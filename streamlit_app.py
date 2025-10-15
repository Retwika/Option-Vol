"""
Streamlit app for Option Strategy Payoff & Greeks Visualizer.

This is the main UI file that uses modular components for calculations and visualizations.
"""

import streamlit as st
from option_strategies import OptionStrategy
from config import apply_page_config, apply_custom_styling, show_header, show_footer
from ui_components import render_sidebar
from strategy_tabs import render_strategy_tabs
from volatility_tabs import render_volatility_tabs


def main():
    """Main application entry point."""
    # Setup page configuration
    apply_page_config()
    apply_custom_styling()
    
    # Render header
    show_header()
    
    # Initialize session state
    if "strategy" not in st.session_state:
        st.session_state.strategy = OptionStrategy()
    
    # Render sidebar and get parameters
    sidebar_params = render_sidebar(st.session_state.strategy)
    
    # Create main tabs
    main_tab1, main_tab2 = st.tabs([
        "📊 Option Strategy Payoff & Greeks", 
        "🌐 Volatility Surface & Mispricing Analysis"
    ])
    
    # Render strategy tabs
    with main_tab1:
        render_strategy_tabs(
            st.session_state.strategy, 
            sidebar_params['current_price'], 
            sidebar_params['T'], 
            sidebar_params['r'], 
            sidebar_params['iv'],
            sidebar_params  # Pass all sidebar params for currency formatting
        )
    
    # Render volatility tabs
    with main_tab2:
        render_volatility_tabs(sidebar_params)
    
    # Render footer
    show_footer()


if __name__ == "__main__":
    main()
