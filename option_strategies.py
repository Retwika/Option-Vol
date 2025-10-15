"""
Option strategy management and calculations.

This module handles option legs, strategy combinations, and payoff calculations.
"""

import numpy as np
from typing import List, Dict, Tuple
from black_scholes import calculate_greeks_vectorized


class OptionLeg:
    """Represents a single option leg in a strategy."""
    
    def __init__(self, option_type: str, action: str, strike_price: float, 
                 premium: float, quantity: int):
        """
        Initialize an option leg.
        
        Args:
            option_type: "Call" or "Put"
            action: "Buy" or "Sell"
            strike_price: Strike price of the option
            premium: Option premium per contract
            quantity: Number of contracts
        """
        self.option_type = option_type
        self.action = action
        self.strike_price = strike_price
        self.premium = premium
        self.quantity = quantity
    
    def calculate_payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate payoff for this leg across a range of underlying prices.
        
        Args:
            prices: Array of underlying prices
            
        Returns:
            Array of payoffs for this leg
        """
        # Calculate intrinsic value
        if self.option_type == "Call":
            intrinsic = np.maximum(prices - self.strike_price, 0)
        else:  # Put
            intrinsic = np.maximum(self.strike_price - prices, 0)
        
        # Calculate payoff considering premium and action
        if self.action == "Buy":
            payoff = intrinsic - self.premium
        else:  # Sell
            payoff = self.premium - intrinsic
        
        return self.quantity * payoff
    
    def calculate_greeks(self, prices: np.ndarray, T: float, r: float, 
                        sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Greeks for this leg across a range of underlying prices.
        
        Args:
            prices: Array of underlying prices
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of arrays (delta, gamma, theta, vega) for this leg
        """
        delta, gamma, theta, vega = calculate_greeks_vectorized(
            prices, self.strike_price, T, r, sigma, self.option_type
        )
        
        # Adjust for action (sell = negative Greeks, except theta)
        if self.action == "Sell":
            delta = -delta
            gamma = -gamma
            # Theta: When you sell an option, you benefit from time decay
            # So theta should be positive for short positions (opposite of long)
            theta = -theta
            vega = -vega
        
        # Scale by quantity
        return (self.quantity * delta, self.quantity * gamma, 
                self.quantity * theta, self.quantity * vega)
    
    def __str__(self) -> str:
        """String representation of the option leg."""
        return (f"{self.action} {self.quantity} {self.option_type} "
                f"@ {self.strike_price} | Premium: {self.premium}")


class OptionStrategy:
    """Manages a collection of option legs forming a strategy."""
    
    def __init__(self):
        """Initialize an empty strategy."""
        self.legs: List[OptionLeg] = []
    
    def add_leg(self, option_type: str, action: str, strike_price: float, 
                premium: float, quantity: int) -> None:
        """Add a new leg to the strategy."""
        leg = OptionLeg(option_type, action, strike_price, premium, quantity)
        self.legs.append(leg)
    
    def remove_leg(self, index: int) -> None:
        """Remove a leg from the strategy by index."""
        if 0 <= index < len(self.legs):
            self.legs.pop(index)
    
    def clear_legs(self) -> None:
        """Remove all legs from the strategy."""
        self.legs.clear()
    
    def get_price_range(self, range_factor: float = 0.5) -> np.ndarray:
        """
        Generate a price range based on the strategy's strike prices.
        
        Args:
            range_factor: Factor to extend the range beyond min/max strikes
            
        Returns:
            Array of prices for analysis
        """
        if not self.legs:
            return np.linspace(50, 150, 500)
        
        strikes = [leg.strike_price for leg in self.legs]
        min_strike = min(strikes)
        max_strike = max(strikes)
        
        price_min = min_strike * (1 - range_factor)
        price_max = max_strike * (1 + range_factor)
        
        return np.linspace(price_min, price_max, 500)
    
    def calculate_total_payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate total payoff for the entire strategy.
        
        Args:
            prices: Array of underlying prices
            
        Returns:
            Array of total payoffs
        """
        if not self.legs:
            return np.zeros_like(prices)
        
        total_payoff = np.zeros_like(prices)
        for leg in self.legs:
            total_payoff += leg.calculate_payoff(prices)
        
        return total_payoff
    
    def calculate_total_greeks(self, prices: np.ndarray, T: float, r: float, 
                              sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate total Greeks for the entire strategy.
        
        Args:
            prices: Array of underlying prices
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of arrays (total_delta, total_gamma, total_theta, total_vega)
        """
        if not self.legs:
            zeros = np.zeros_like(prices)
            return zeros, zeros, zeros, zeros
        
        total_delta = np.zeros_like(prices)
        total_gamma = np.zeros_like(prices)
        total_theta = np.zeros_like(prices)
        total_vega = np.zeros_like(prices)
        
        for leg in self.legs:
            delta, gamma, theta, vega = leg.calculate_greeks(prices, T, r, sigma)
            total_delta += delta
            total_gamma += gamma
            total_theta += theta
            total_vega += vega
        
        return total_delta, total_gamma, total_theta, total_vega
    
    def calculate_theta_surface(self, prices: np.ndarray, time_range: np.ndarray, 
                               r: float, sigma: float) -> np.ndarray:
        """
        Calculate theta surface for the strategy across time and price.
        
        Args:
            prices: Array of underlying prices
            time_range: Array of time to expiration values (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            2D array of theta values [time, price]
        """
        theta_surface = []
        
        for T in time_range:
            row = []
            for price in prices:
                total_theta = 0
                for leg in self.legs:
                    _, _, theta, _ = leg.calculate_greeks(
                        np.array([price]), T, r, sigma
                    )
                    total_theta += theta[0]
                row.append(total_theta)
            theta_surface.append(row)
        
        return np.array(theta_surface)
    
    def get_strategy_summary(self) -> List[Dict]:
        """
        Get a summary of all legs in the strategy.
        
        Returns:
            List of dictionaries containing leg information
        """
        return [
            {
                "option_type": leg.option_type,
                "action": leg.action,
                "strike_price": leg.strike_price,
                "premium": leg.premium,
                "quantity": leg.quantity,
                "description": str(leg)
            }
            for leg in self.legs
        ]
    
    def __len__(self) -> int:
        """Return the number of legs in the strategy."""
        return len(self.legs)
    
    def __bool__(self) -> bool:
        """Return True if strategy has legs."""
        return len(self.legs) > 0
