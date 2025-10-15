"""
Configuration and styling for the Streamlit app.
"""

import streamlit as st

# App configuration
APP_CONFIG = {
    "page_title": "Option Strategy + Greeks Visualizer",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom CSS for the app
CUSTOM_CSS = '''
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #ff4b4b; 
        color: white; 
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border: none;
    }
    .st-expanderHeader {
        font-weight: bold; 
        color: #2c3e50;
    }
    .stPlotlyChart {
        background-color: #fff; 
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMarkdown {margin-bottom: 0.5rem;}
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
'''

# Display Settings
DISPLAY_SETTINGS = {
    'iv_decimal_places': 2,   # Decimal places for IV display
    'price_decimal_places': 2, # Decimal places for price display
    'percentage_decimal_places': 1, # Decimal places for percentage display
}

# Plot Settings
PLOT_SETTINGS = {
    'surface_height': 600,    # 3D surface plot height
    'heatmap_height': 500,    # Heatmap height
    'smile_height': 400,      # Volatility smile height
    'min_surface_points': 4,  # Minimum points for 3D surface
    'surface_colorscale': 'Viridis',  # 3D surface colorscale
    'heatmap_colorscale': 'Viridis',   # Heatmap colorscale
}

# Cache TTL Settings (in seconds)
CACHE_TTL = {
    'data_fetch_ttl': 60,     # 1 minute for live data
}

def apply_page_config():
    """Apply page configuration."""
    st.set_page_config(**APP_CONFIG)

def apply_custom_styling():
    """Apply custom CSS styling."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def show_header():
    """Display the main header."""
    st.title("📊 Advanced Options Analysis Platform")
    st.markdown("*Build complex option strategies and analyze market opportunities with real-time data*")
    st.markdown("---")

def show_footer():
    """Display the footer."""
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: #888; font-size: 0.9rem;'>"
        "Made with ❤️ using Streamlit, Plotly, and NumPy | 2025 | "
        "<a href='#' style='color: #888;'>View Source Code</a>"
        "</div>", 
        unsafe_allow_html=True
    )
