"""
Black-Scholes option pricing and Greeks calculations.

This module provides functions to calculate option prices and Greeks
using the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str) -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "Call" or "Put"
    
    Returns:
        Option price
    """
    if T <= 0:
        # Handle edge case for expiry
        if option_type == "Call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str):
    """
    Calculate Black-Scholes Greeks.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "Call" or "Put"
    
    Returns:
        Tuple of (delta, gamma, theta, vega)
    """
    if T <= 0:
        # Handle edge case for expiry
        if option_type == "Call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return delta, 0.0, 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common terms
    phi_d1 = norm.pdf(d1)  # Standard normal PDF at d1
    cdf_d1 = norm.cdf(d1)  # Standard normal CDF at d1
    cdf_d2 = norm.cdf(d2)  # Standard normal CDF at d2
    
    # Delta
    if option_type == "Call":
        delta = cdf_d1
    else:  # Put
        delta = cdf_d1 - 1.0  # Equivalent to -norm.cdf(-d1)
    
    # Gamma (same for calls and puts)
    gamma = phi_d1 / (S * sigma * np.sqrt(T))
    
    # Theta (per day, not per year)
    if option_type == "Call":
        theta = ((-S * phi_d1 * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * cdf_d2) / 365
    else:  # Put
        theta = ((-S * phi_d1 * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (same for calls and puts, per 1% change in volatility)
    vega = S * phi_d1 * np.sqrt(T) / 100
    
    return delta, gamma, theta, vega


def calculate_implied_volatility(market_price: float, S: float, K: float, T: float, 
                               r: float, option_type: str, tolerance: float = 1e-6, 
                               max_iterations: int = 100) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        market_price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        option_type: "Call" or "Put"
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
    
    Returns:
        Implied volatility
    """
    # Initial guess
    sigma = 0.2
    
    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = black_scholes_greeks(S, K, T, r, sigma, option_type)[3] * 100  # Convert back to per unit
        
        if abs(vega) < tolerance:
            break
            
        price_diff = price - market_price
        
        if abs(price_diff) < tolerance:
            break
            
        # Newton-Raphson update
        sigma = sigma - price_diff / vega
        
        # Ensure sigma stays positive
        sigma = max(sigma, 0.001)
    
    return sigma


def calculate_greeks_vectorized(prices: np.ndarray, K: float, T: float, r: float, 
                              sigma: float, option_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Greeks for an array of underlying prices (vectorized for performance).
    
    Args:
        prices: Array of underlying prices
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "Call" or "Put"
    
    Returns:
        Tuple of arrays (delta, gamma, theta, vega)
    """
    if T <= 0:
        # Handle edge case for expiry
        if option_type == "Call":
            delta = (prices > K).astype(float)
        else:
            delta = -(prices < K).astype(float)
        return delta, np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
    
    # Avoid numerical issues with very small prices
    prices = np.maximum(prices, 1e-6)
    
    d1 = (np.log(prices / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common terms
    phi_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    # Delta
    if option_type == "Call":
        delta = cdf_d1
    else:  # Put
        delta = cdf_d1 - 1.0
    
    # Gamma (same for calls and puts)
    gamma = phi_d1 / (prices * sigma * np.sqrt(T))
    
    # Theta (per day)
    if option_type == "Call":
        theta = ((-prices * phi_d1 * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * cdf_d2) / 365
    else:  # Put
        theta = ((-prices * phi_d1 * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (same for calls and puts, per 1% change)
    vega = prices * phi_d1 * np.sqrt(T) / 100
    
    return delta, gamma, theta, vega
