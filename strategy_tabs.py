"""
Strategy analysis tabs for the Streamlit app.
"""

import streamlit as st
import numpy as np
from visualization import (
    create_payoff_chart, create_greeks_chart, create_combined_greeks_charts,
    create_theta_surface, create_strategy_comparison_chart, 
    create_profit_loss_zones_chart, create_risk_metrics_table
)
from ui_components import render_strategy_input_form, render_strategy_templates, render_welcome_message

def render_combined_strategy_analysis(strategy, current_price, T, r, iv, currency_symbol="$"):
    """Render the combined strategy analysis tab."""
    st.markdown("## 📈 Combined Strategy Analysis")
    
    # Calculate strategy data
    prices = strategy.get_price_range()
    total_payoff = strategy.calculate_total_payoff(prices)
    total_delta, total_gamma, total_theta, total_vega = strategy.calculate_total_greeks(
        prices, T, r, iv
    )
    
    # Main charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Payoff chart with profit/loss zones
        pnl_chart = create_profit_loss_zones_chart(prices, total_payoff, currency_symbol)
        st.plotly_chart(pnl_chart, use_container_width=True)
        
    with chart_col2:
        # Strategy comparison chart
        comparison_chart = create_strategy_comparison_chart(
            strategy, prices, T, r, iv, currency_symbol
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Greeks charts
    st.markdown("#### 📊 Greeks Analysis")
    greeks_figures = create_combined_greeks_charts(
        prices, total_delta, total_gamma, total_theta, total_vega, currency_symbol
    )
    
    greeks_cols = st.columns(2)
    for i, fig in enumerate(greeks_figures):
        with greeks_cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)
    
    # Theta surface
    st.markdown("#### ⏳ Theta Decay Surface")
    time_grid = np.linspace(1/365, 60/365, 30)  # 1 to 60 days
    theta_surface_data = strategy.calculate_theta_surface(
        prices[::10], time_grid, r, iv  # Use fewer price points for performance
    )
    
    theta_fig = create_theta_surface(
        prices[::10], time_grid * 365, theta_surface_data,
        "Theta Surface - Combined Strategy"
    )
    st.plotly_chart(theta_fig, use_container_width=True)

def render_individual_leg_analysis(strategy, T, r, iv, currency_symbol="$"):
    """Render the individual leg analysis tab."""
    st.markdown("## 🔍 Individual Leg Analysis")
    
    if len(strategy.legs) > 1:
        # Calculate strategy data
        prices = strategy.get_price_range()
        time_grid = np.linspace(1/365, 60/365, 30)
        
        # Leg selector
        leg_options = [
            f"Leg {i+1}: {leg.option_type} @ ${leg.strike_price}" 
            for i, leg in enumerate(strategy.legs)
        ]
        selected_leg_idx = st.selectbox(
            "Select Leg to Analyze", 
            options=list(range(len(strategy.legs))),
            format_func=lambda i: leg_options[i]
        )
        
        leg = strategy.legs[selected_leg_idx]
        st.markdown(f"#### Analysis for {leg}")
        
        # Individual leg calculations
        leg_payoff = leg.calculate_payoff(prices)
        leg_delta, leg_gamma, leg_theta, leg_vega = leg.calculate_greeks(prices, T, r, iv)
        
        # Individual leg charts
        individual_cols = st.columns(2)
        
        with individual_cols[0]:
            payoff_fig = create_payoff_chart(prices, leg_payoff, f"Payoff - Leg {selected_leg_idx+1}")
            st.plotly_chart(payoff_fig, use_container_width=True)
            
            gamma_fig = create_greeks_chart(prices, leg_gamma, "Gamma", f"Gamma - Leg {selected_leg_idx+1}")
            st.plotly_chart(gamma_fig, use_container_width=True)
        
        with individual_cols[1]:
            delta_fig = create_greeks_chart(prices, leg_delta, "Delta", f"Delta - Leg {selected_leg_idx+1}")
            st.plotly_chart(delta_fig, use_container_width=True)
            
            theta_fig = create_greeks_chart(prices, leg_theta, "Theta", f"Theta - Leg {selected_leg_idx+1}")
            st.plotly_chart(theta_fig, use_container_width=True)
        
        # Individual theta surface
        st.markdown(f"#### ⏳ Theta Surface - Leg {selected_leg_idx+1}")
        individual_theta_surface = []
        for T2 in time_grid:
            row = []
            for price in prices[::10]:
                _, _, theta, _ = leg.calculate_greeks(np.array([price]), T2, r, iv)
                row.append(theta[0])
            individual_theta_surface.append(row)
        
        individual_theta_fig = create_theta_surface(
            prices[::10], time_grid * 365, np.array(individual_theta_surface),
            f"Theta Surface - Leg {selected_leg_idx+1}"
        )
        st.plotly_chart(individual_theta_fig, use_container_width=True)
    else:
        st.info("Add more legs to compare individual performances.")

def render_advanced_analysis(strategy, current_price, T, r, iv, currency_symbol="$"):
    """Render the advanced analysis tab."""
    st.markdown("## 📐 Advanced Analysis")
    
    # Scenario analysis
    st.markdown("### 🎯 Scenario Analysis")
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        st.markdown("**Price Scenarios**")
        price_changes = [-20, -10, -5, 0, 5, 10, 20]
        scenario_prices = [current_price * (1 + change/100) for change in price_changes]
        scenario_payoffs = strategy.calculate_total_payoff(np.array(scenario_prices))
        
        scenario_df = {
            "Price Change (%)": price_changes,
            "Underlying Price": [f"${p:.2f}" for p in scenario_prices],
            "Strategy P&L": [f"${pnl:.2f}" for pnl in scenario_payoffs]
        }
        st.dataframe(scenario_df, use_container_width=True)
    
    with scenario_col2:
        st.markdown("**Time Decay Analysis**")
        time_scenarios = [30, 21, 14, 7, 3, 1]  # Days
        current_prices_array = np.array([current_price])
        
        time_decay_data = []
        for days in time_scenarios:
            T_scenario = days / 365
            _, _, theta, _ = strategy.calculate_total_greeks(
                current_prices_array, T_scenario, r, iv
            )
            time_decay_data.append({
                "Days to Expiry": days,
                "Daily Theta": f"${theta[0]:.4f}"
            })
        
        st.dataframe(time_decay_data, use_container_width=True)
    
    # Volatility sensitivity
    st.markdown("### 📈 Volatility Sensitivity")
    vol_scenarios = np.arange(0.1, 0.5, 0.05)
    vol_pnl = []
    
    for vol in vol_scenarios:
        _, _, _, vega = strategy.calculate_total_greeks(
            np.array([current_price]), T, r, float(vol)
        )
        vol_pnl.append(vega[0])
    
    vol_fig = create_greeks_chart(
        vol_scenarios * 100, np.array(vol_pnl), "Vega", 
        "Vega vs Implied Volatility"
    )
    vol_fig.update_layout(xaxis_title="Implied Volatility (%)")
    st.plotly_chart(vol_fig, use_container_width=True)

def render_strategy_tabs(strategy, current_price, T, r, iv, sidebar_params=None):
    """Render all strategy analysis tabs."""
    # Determine currency symbol based on data source
    currency_symbol = "$" if sidebar_params and sidebar_params.get('data_source') == "US Options (Yahoo Finance)" else "₹"
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Option leg input section
        render_strategy_input_form(strategy, current_price, currency_symbol)
    
    with col2:
        if strategy:
            # Risk metrics
            st.markdown("### 📊 Risk Metrics")
            risk_table = create_risk_metrics_table(strategy, current_price, T, r, iv, currency_symbol)
            st.plotly_chart(risk_table, use_container_width=True)
    
    # Display current strategy analysis or welcome message
    if strategy:
        # Strategy summary
        st.markdown("---")
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
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Strategy Analysis", "🔍 Individual Legs", "📐 Advanced Analysis"])
        
        with tab1:
            render_combined_strategy_analysis(strategy, current_price, T, r, iv, currency_symbol)
        
        with tab2:
            render_individual_leg_analysis(strategy, T, r, iv, currency_symbol)
        
        with tab3:
            render_advanced_analysis(strategy, current_price, T, r, iv, currency_symbol)
    else:
        # Welcome message and templates when no strategy is built
        render_welcome_message()
        render_strategy_templates(strategy, current_price, currency_symbol)
