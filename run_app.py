#!/usr/bin/env python3
"""
Launch script for the Option Strategy Visualizer.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application."""
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("Error: streamlit_app.py not found in current directory")
        print("Please run this script from the opt_streamlit directory")
        return 1
    
    # Check if requirements are installed
    try:
        import streamlit
        import numpy
        import plotly
        import scipy
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return 1
    
    # Launch Streamlit
    print("🚀 Launching Option Strategy Visualizer...")
    print("📊 Building your option strategy analysis tool...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to launch Streamlit application")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
