# 📊 Advanced Options Analysis Platform

A comprehensive Streamlit application for building and analyzing complex option strategies with real-time payoff calculations, advanced Greeks visualization, and professional volatility surface analysis.

## 🚀 Key Features

### **Option Strategy Analysis**
- 📊 **Interactive Strategy Building**: Add multiple option legs with different strikes, types, and quantities
- 📈 **Real-time Payoff Analysis**: Live P&L calculations across price ranges with profit/loss zones
- 🧮 **Advanced Greeks**: Delta, Gamma, Theta, Vega calculations for individual legs and combined strategies
- ⏳ **3D Theta Surfaces**: Visualize time decay across price and time dimensions
- 🎯 **Scenario Analysis**: Test strategies under different price and volatility scenarios
- 📐 **Risk Metrics**: Key risk indicators and breakeven analysis

### **Volatility Surface & Arbitrage Analysis**
- 🌐 **3D Volatility Surfaces**: Interactive implied volatility visualization
- 🔍 **Arbitrage Detection**: Automated scanning for calendar, butterfly, and vertical arbitrages
- 📊 **Put-Call Parity Analysis**: Identify mispricing opportunities
- 📈 **SVI Model Fitting**: Professional volatility smile modeling
- 🌡️ **Volatility Heatmaps**: IV spread analysis across strikes and expiries

### **Live Data Integration**
- 🇮🇳 **NIFTY Options**: Real-time NSE options data via `nsepython`
- 🇺🇸 **US Options**: Live Yahoo Finance data for popular symbols (AAPL, SPY, TSLA, etc.)
- 🔄 **Auto-refresh**: Cached data with freshness indicators
- 💱 **Multi-currency**: Support for ₹ (INR) and $ (USD) display

### **Professional Features**
- 🎨 **Modern UI**: Clean, responsive interface with interactive charts
- ⚡ **Performance Optimized**: Cached calculations and efficient data processing
- 📱 **Mobile Responsive**: Works on desktop and mobile devices
- 🛡️ **Error Handling**: Robust error handling and fallback mechanisms

## 📁 Project Structure

```
opt_streamlit/
├── streamlit_app.py          # Main Streamlit application entry point
├── run_app.py               # Launch script with dependency checking
├── config.py                # Configuration, styling, and settings
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── GREEKS_FIXES.md         # Technical fixes and improvements log
│
├── Strategy Analysis/
│   ├── option_strategies.py    # Core strategy logic and calculations
│   ├── strategy_tabs.py        # Strategy analysis UI components
│   ├── black_scholes.py        # Black-Scholes pricing and Greeks
│   └── ui_components.py        # Reusable UI components and forms
│
├── Volatility Analysis/
│   ├── volatility_tabs.py      # Advanced volatility surface analysis
│   ├── svi_model.py           # SVI volatility model implementation
│   ├── nifty_data.py          # Live data fetching (NSE & Yahoo Finance)
│   └── data_utils.py          # Data processing and caching utilities
│
└── Visualization/
    ├── visualization.py        # Strategy payoff and Greeks charts
    └── plots.py               # Volatility surface and 3D plotting
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download** this repository
2. **Navigate** to the project directory:
   ```bash
   cd opt_streamlit
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Launch the Application

**Method 1: Using the launch script (Recommended)**
```bash
python run_app.py
```

**Method 2: Direct Streamlit command**
```bash
streamlit run streamlit_app.py
```

4. **Open your browser** to the provided URL (typically `http://localhost:8501`)

## 📖 How to Use

### **Option Strategy Analysis**

1. **Set Global Parameters** (sidebar):
   - Choose data source: Manual Input, NIFTY Live Data, or US Options
   - Set current underlying price, volatility, risk-free rate, and time to expiry

2. **Build Your Strategy**:
   - Add option legs using the form (calls/puts, buy/sell, strikes, premiums)
   - Use quick templates for common strategies (Long Call, Long Put, Long Straddle)
   - Remove individual legs as needed

3. **Analyze Results** across three tabs:
   - **Strategy Analysis**: Combined payoff, Greeks, and 3D theta surfaces
   - **Individual Legs**: Breakdown of each option leg's contribution
   - **Advanced Analysis**: Scenario testing and volatility sensitivity

### **Volatility Surface Analysis**

1. **Select Data Source**:
   - **NIFTY**: Automatic live data from NSE
   - **US Options**: Choose from popular symbols or enter custom symbol

2. **Analyze Volatility**:
   - View 3D volatility surfaces and heatmaps
   - Examine volatility smiles for specific expiries
   - Identify arbitrage opportunities automatically

3. **Arbitrage Scanner**:
   - Calendar arbitrage (time spreads)
   - Butterfly arbitrage (convexity violations)
   - Vertical arbitrage (put-call parity violations)

## 🧮 Core Components

### **Strategy Management (`option_strategies.py`)**
- `OptionLeg` class: Individual option positions with payoff and Greeks calculations
- `OptionStrategy` class: Collection of legs with combined analysis
- Theta surface generation across time and price dimensions

### **Black-Scholes Engine (`black_scholes.py`)**
- Pure Black-Scholes pricing functions with edge case handling
- Vectorized Greeks calculations (Delta, Gamma, Theta, Vega) for performance
- Implied volatility calculations using Newton-Raphson method

### **Live Data Integration (`nifty_data.py`)**
- NSE NIFTY options via `nsepython` with multiple expiry support
- US options via `yfinance` with validation and error handling
- Volatility surface metrics calculation and caching

### **Visualization Suite (`visualization.py`, `plots.py`)**
- Interactive Plotly charts with profit/loss zone highlighting
- 3D surfaces for theta decay and volatility analysis
- Smart scaling for Greeks (log scale for Gamma/Vega when appropriate)
- Risk metrics tables and scenario analysis charts

### **Advanced Volatility Analysis (`volatility_tabs.py`)**
- 3D volatility surface construction with interpolation
- Automated arbitrage detection using no-arbitrage conditions
- Put-call IV spread analysis and mispricing identification
- SVI model integration for professional volatility fitting

## 📊 Example Strategies & Use Cases

### **Popular Option Strategies**
- **Long Call/Put**: Basic directional plays with unlimited upside
- **Long Straddle**: Volatility plays for earnings or events
- **Iron Condor**: Range-bound strategies with limited risk/reward
- **Butterfly Spreads**: Profit from low volatility around specific price
- **Calendar Spreads**: Benefit from time decay differences

### **Real-World Applications**
- **Portfolio Hedging**: Protect existing positions with puts
- **Income Generation**: Sell covered calls or cash-secured puts  
- **Volatility Trading**: Profit from IV changes using straddles/strangles
- **Arbitrage Opportunities**: Identify mispriced options automatically
- **Risk Management**: Visualize maximum loss and breakeven points

## 🛠️ Technical Features

### **Performance Optimizations**
- ✅ Cached data fetching with TTL (Time To Live) settings
- ✅ Vectorized NumPy calculations for Greeks across price ranges
- ✅ Efficient DataFrame operations with pandas
- ✅ Smart chart rendering with data point limitations

### **Error Handling & Robustness**
- ✅ Graceful fallbacks when live data is unavailable
- ✅ Input validation for option parameters
- ✅ Numerical stability for extreme market conditions
- ✅ Edge case handling (zero time to expiry, negative prices)

### **Data Sources & Reliability**
- ✅ Primary: NSE (nsepython) for Indian markets
- ✅ Secondary: Yahoo Finance (yfinance) for global markets
- ✅ Fallback: Manual input mode when live data fails
- ✅ Data validation and cleaning pipelines

## 🧪 Advanced Features

### **SVI Volatility Model**
The platform includes a complete SVI (Stochastic Volatility Inspired) implementation:
- Fits parametric volatility smiles to market data
- Ensures no-arbitrage conditions are satisfied
- Provides smooth extrapolation beyond observed strikes
- Professional-grade model used by quantitative traders

### **Arbitrage Detection Engine**
Automated scanning for three types of arbitrage:
1. **Calendar Arbitrage**: Longer expiry options cheaper than shorter ones
2. **Butterfly Arbitrage**: Convexity violations in option prices
3. **Vertical Arbitrage**: Put-call parity violations across strikes

### **3D Visualization Suite**
- Interactive 3D volatility surfaces with time and strike dimensions
- Theta decay surfaces showing time value erosion
- Real-time rotation and zooming capabilities
- Professional colorscales and hover information

## 📈 Dependencies & Requirements

### **Core Libraries**
- **Streamlit** (>=1.28.0): Web application framework with caching
- **NumPy** (>=1.24.0): Numerical computations and vectorized operations
- **Plotly** (>=5.15.0): Interactive charts and 3D visualizations  
- **SciPy** (>=1.10.0): Statistical functions and optimization
- **Pandas** (>=2.0.0): Data manipulation and analysis

### **Data Sources**
- **nsepython** (>=1.0.0): NSE India options data
- **yfinance** (>=0.2.18): Yahoo Finance global markets data

### **System Requirements**
- Python 3.8+ (tested on 3.8, 3.9, 3.10, 3.11)
- 4GB+ RAM recommended for large datasets
- Modern web browser with JavaScript enabled
- Internet connection for live data feeds

## 🐛 Troubleshooting

### **Common Issues**

**"No data found" for US symbols:**
- Verify the symbol exists and has active options
- Try popular symbols first (SPY, AAPL, MSFT)
- Check if markets are open (data may be delayed after hours)

**NIFTY data loading errors:**
- NSE servers may be temporarily unavailable
- Switch to manual input mode as fallback
- Check internet connection and firewall settings

**Charts not displaying:**
- Ensure JavaScript is enabled in your browser
- Try refreshing the page or clearing browser cache
- Check browser console for error messages

**Performance issues:**
- Reduce the number of option legs in complex strategies
- Use fewer expiry dates in volatility analysis
- Close other browser tabs to free up memory

### **Debug Mode**
Enable debug mode in the sidebar to see:
- Detailed error messages and stack traces
- Data loading progress and validation steps
- Calculation timing and performance metrics
- Raw data inspection for troubleshooting

## 🤝 Contributing

We welcome contributions! Areas for improvement:

### **Priority Enhancements**
- Additional exotic option types (barriers, digitals)
- More sophisticated volatility models (Heston, SABR)
- Portfolio-level Greeks and risk aggregation
- Historical volatility analysis and backtesting
- Options flow analysis and unusual activity detection

### **Technical Improvements**
- Real-time streaming data connections
- Database integration for historical storage
- API endpoints for programmatic access
- Mobile app development
- Performance optimization for large datasets

### **How to Contribute**
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request with detailed description
5. Follow existing code style and documentation patterns

## 📄 License & Disclaimer

### **License**
This project is open source and available under the **MIT License**.

### **Important Disclaimer**
⚠️ **This application is for educational and research purposes only.**

- Not intended as investment advice or recommendations
- Options trading involves substantial risk of loss
- Past performance does not guarantee future results  
- Consult with qualified financial advisors before trading
- Verify all calculations independently before making trading decisions
- The authors assume no liability for trading losses

### **Data Disclaimer**
- Market data may be delayed or inaccurate
- Third-party data providers have their own terms of service
- Always verify prices with official exchanges before trading
- Real-time data may require separate subscriptions

---

## 🙏 Acknowledgments

**Built with:**
- [Streamlit](https://streamlit.io/) - Amazing web app framework
- [Plotly](https://plotly.com/) - Interactive visualization library
- [NumPy/SciPy](https://numpy.org/) - Scientific computing foundation
- [nsepython](https://github.com/jugaad-py/nsepython) - NSE data access
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API

**Inspired by:**
- Professional options trading platforms
- Quantitative finance research and academic literature
- Open source derivatives pricing libraries
- Financial engineering best practices

---

**Happy Trading! 📈📊**

*Made with ❤️ for the options trading and quantitative finance community*
