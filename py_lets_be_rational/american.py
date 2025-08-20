# -*- coding: utf-8 -*-
"""
py_lets_be_rational.american
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

American option pricing on commodity futures using Black-76 as foundation.
Implements Barone-Adesi & Whaley and Bjerksund-Stensland approximations.

:copyright: © 2025 Extended Implementation  
:license: MIT, see LICENSE for more details.

About American Options:
~~~~~~~~~~~~~~~~~~~~~~

Implements several approximation methods:
1. Barone-Adesi & Whaley (1987) - Quadratic approximation
2. Bjerksund-Stensland (1993) - Exercise boundary approximation  
3. Black's approximation for pseudo-dividends

All methods use Black-76 as the European baseline.
"""

# -----------------------------------------------------------------------------
# IMPORTS

# Standard library imports
from math import log, sqrt, exp, fabs
import warnings

# Related third party imports
import numpy as np

# Local application/library specific imports
from py_lets_be_rational.constants import DBL_EPSILON, SQRT_DBL_EPSILON
from py_lets_be_rational.exceptions import VolatilityValueException
from py_lets_be_rational.numba_helper import maybe_jit
from py_lets_be_rational.normaldistribution import norm_cdf
from py_lets_be_rational.lets_be_rational import black

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (can be JIT compiled)

@maybe_jit(cache=True, nopython=True)
def _d1_calc(F, K, T, r, sigma):
    """Calculate d1 for Black-76."""
    return (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))

@maybe_jit(cache=True, nopython=True)
def _q1_calc(r, convenience_yield, sigma):
    """Calculate q1 for put early exercise."""
    discriminant = (convenience_yield - 0.5 * sigma**2)**2 + 2 * r * sigma**2
    return 0.5 - convenience_yield / sigma**2 - sqrt(discriminant) / sigma**2

@maybe_jit(cache=True, nopython=True)
def _q2_calc(r, convenience_yield, sigma):
    """Calculate q2 for call early exercise."""
    discriminant = (convenience_yield - 0.5 * sigma**2)**2 + 2 * r * sigma**2
    return 0.5 - convenience_yield / sigma**2 + sqrt(discriminant) / sigma**2

# -----------------------------------------------------------------------------
# CORE AMERICAN OPTION PRICER CLASS

class AmericanBlack76Pricer:
    """
    American option pricing on commodity futures using Black-76 as foundation.
    
    Integrates with py_lets_be_rational for consistent European option pricing.
    """
    
    def __init__(self):
        self.eps = SQRT_DBL_EPSILON
        self.max_iterations = 100
        
    def black76_european(self, F, K, T, r, sigma, option_type='call'):
        """
        European option pricing using py_lets_be_rational's black function.
        
        Parameters:
        -----------
        F : float
            Current futures price
        K : float  
            Strike price
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float : European option price
        """
        if sigma < 0:
            raise VolatilityValueException()
            
        if T <= 0:
            if option_type == 'call':
                return max(F - K, 0) * exp(-r * T) if T == 0 else 0
            else:
                return max(K - F, 0) * exp(-r * T) if T == 0 else 0
        
        # Use py_lets_be_rational's black function with discount factor
        q = 1 if option_type == 'call' else -1
        forward_value = black(F, K, sigma, T, q)
        return forward_value * exp(-r * T)
    
    def _d1(self, F, K, T, r, sigma):
        """Calculate d1 for Black-76."""
        return _d1_calc(F, K, T, r, sigma)
    
    def _q1_factor(self, r, convenience_yield, sigma):
        """Calculate q1 for put early exercise."""
        return _q1_calc(r, convenience_yield, sigma)
    
    def _q2_factor(self, r, convenience_yield, sigma):
        """Calculate q2 for call early exercise."""
        return _q2_calc(r, convenience_yield, sigma)
    
    def barone_adesi_whaley_call(self, F, K, T, r, sigma, convenience_yield=0.0):
        """
        Barone-Adesi & Whaley approximation for American call on futures.
        
        Parameters:
        -----------
        convenience_yield : float
            Convenience yield of the commodity (acts like dividend)
        """
        # European option value using py_lets_be_rational
        european_call = self.black76_european(F, K, T, r, sigma, 'call')
        
        # If convenience yield is zero or negative, American call = European call
        if convenience_yield <= self.eps:
            return european_call
        
        # Calculate critical price for early exercise
        S_star = self._critical_price_call(F, K, T, r, sigma, convenience_yield)
        
        if F <= S_star:
            return european_call
        
        # Early exercise premium
        q2 = self._q2_factor(r, convenience_yield, sigma)
        d1_star = self._d1(S_star, K, T, r, sigma)
        A2 = (S_star / q2) * (1 - exp(-convenience_yield * T) * norm_cdf(d1_star))
        
        early_exercise_premium = A2 * (F / S_star)**q2
        
        return european_call + early_exercise_premium
    
    def barone_adesi_whaley_put(self, F, K, T, r, sigma, convenience_yield=0.0):
        """
        Barone-Adesi & Whaley approximation for American put on futures.
        """
        # European option value using py_lets_be_rational
        european_put = self.black76_european(F, K, T, r, sigma, 'put')
        
        # Calculate critical price for early exercise
        S_star = self._critical_price_put(F, K, T, r, sigma, convenience_yield)
        
        if F >= S_star:
            return european_put
        
        # Early exercise premium
        q1 = self._q1_factor(r, convenience_yield, sigma)
        d1_star = self._d1(S_star, K, T, r, sigma)
        A1 = -(S_star / q1) * (1 - exp(-convenience_yield * T) * norm_cdf(-d1_star))
        
        early_exercise_premium = A1 * (F / S_star)**q1
        
        return european_put + early_exercise_premium
    
    def bjerksund_stensland_call(self, F, K, T, r, sigma, convenience_yield=0.0):
        """
        Bjerksund-Stensland (1993) approximation for American call.
        """
        if convenience_yield <= self.eps:
            return self.black76_european(F, K, T, r, sigma, 'call')
        
        # Model parameters
        b = -convenience_yield
        sigma_sq = sigma * sigma
        
        # Calculate beta (quadratic formula solution)
        discriminant = (b/sigma_sq - 0.5)**2 + 2*r/sigma_sq
        if discriminant < 0:
            # Fallback to European if discriminant is negative
            return self.black76_european(F, K, T, r, sigma, 'call')
        
        beta = (0.5 - b/sigma_sq) + sqrt(discriminant)
        
        if beta <= 1:
            # If beta <= 1, American call = European call
            return self.black76_european(F, K, T, r, sigma, 'call')
        
        B_infinity = beta / (beta - 1) * K
        B_0 = max(K, r / convenience_yield * K)
        
        # Check for division by zero
        if abs(B_0 - K) < self.eps:
            # When B_0 ≈ K, the exercise boundary is at the strike
            # This typically means immediate exercise is optimal
            return max(F - K, 0.0)
        
        h_T = -(b * T + 2 * sigma * sqrt(T)) * K / (B_0 - K)
        
        # Prevent numerical overflow in exp(h_T)
        if h_T < -50:  # exp(-50) ≈ 0
            I = B_infinity
        else:
            I = B_0 + (B_infinity - B_0) * (1 - exp(h_T))
        
        # Early exercise check
        if F >= I:
            return F - K
        
        # Prevent negative or zero trigger price
        if I <= K:
            return self.black76_european(F, K, T, r, sigma, 'call')
        
        alpha = (I - K) * I**(-beta)
        
        # Calculate option value components
        d1 = (log(F/K) + (b + sigma_sq/2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        european_part = F * exp((b-r)*T) * norm_cdf(d1) - K * exp(-r*T) * norm_cdf(d2)
        
        # American premium calculation with safety checks
        log_ratio = log(F/I)
        d_alpha_exponent = (log_ratio + (b + (beta-0.5)*sigma_sq) * T) / (sigma * sqrt(T))
        
        american_premium = alpha * (F**beta) * norm_cdf(d_alpha_exponent)
        
        result = european_part + american_premium
        
        # Sanity check: American option should be at least as valuable as European
        european_value = self.black76_european(F, K, T, r, sigma, 'call')
        return max(result, european_value)
    
    def american_option_price(self, F, K, T, r, sigma, option_type='call', 
                             method='barone_adesi_whaley', convenience_yield=0.0):
        """
        Unified interface for American option pricing.
        
        Parameters:
        -----------
        method : str
            'barone_adesi_whaley', 'bjerksund_stensland'
        """
        if method == 'barone_adesi_whaley':
            if option_type == 'call':
                return self.barone_adesi_whaley_call(F, K, T, r, sigma, convenience_yield)
            else:
                return self.barone_adesi_whaley_put(F, K, T, r, sigma, convenience_yield)
        
        elif method == 'bjerksund_stensland':
            if option_type == 'call':
                return self.bjerksund_stensland_call(F, K, T, r, sigma, convenience_yield)
            else:
                # For puts, fall back to Barone-Adesi-Whaley
                return self.barone_adesi_whaley_put(F, K, T, r, sigma, convenience_yield)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _critical_price_call(self, F, K, T, r, sigma, convenience_yield):
        """Find critical futures price for call early exercise using bisection."""
        # Simple bisection search for critical price
        low, high = K, 10 * K
        tolerance = self.eps
        
        for _ in range(self.max_iterations):
            mid = 0.5 * (low + high)
            condition = self._call_exercise_condition(mid, K, T, r, sigma, convenience_yield)
            
            if abs(condition) < tolerance:
                return mid
            
            if condition > 0:
                high = mid
            else:
                low = mid
        
        return high  # Conservative fallback
    
    def _critical_price_put(self, F, K, T, r, sigma, convenience_yield):
        """Find critical futures price for put early exercise using bisection."""
        # For puts, critical price should be between 0 and K
        # Start with a reasonable range
        low = 0.01 * K  # Very low price
        high = K        # Strike price (maximum reasonable early exercise point)
        tolerance = 1e-6
        
        # Sanity check: if convenience yield is very small, return low value
        if convenience_yield < 1e-6:
            return low
        
        # Check boundary conditions first
        try:
            condition_low = self._put_exercise_condition(low, K, T, r, sigma, convenience_yield)
            condition_high = self._put_exercise_condition(high, K, T, r, sigma, convenience_yield)
            
            # If no sign change, early exercise might not be optimal in this range
            if condition_low * condition_high > 0:
                # Check if we should exercise immediately (high convenience yield case)
                if convenience_yield > 0.1:  # 10% threshold
                    # For high convenience yield, critical price approaches strike
                    return 0.95 * K
                else:
                    return low
                    
        except (ValueError, ZeroDivisionError, OverflowError):
            # If calculation fails, assume early exercise not optimal
            return low
        
        # Bisection method
        for iteration in range(self.max_iterations):
            mid = 0.5 * (low + high)
            
            if abs(high - low) < tolerance:
                break
                
            try:
                condition = self._put_exercise_condition(mid, K, T, r, sigma, convenience_yield)
                
                if abs(condition) < tolerance:
                    return mid
                
                if condition * condition_low > 0:
                    low = mid
                    condition_low = condition
                else:
                    high = mid
                    
            except (ValueError, ZeroDivisionError, OverflowError):
                # If calculation fails at midpoint, adjust range
                high = mid
        
        return 0.5 * (low + high)
    
    def _call_exercise_condition(self, S, K, T, r, sigma, convenience_yield):
        """Condition for call early exercise (Barone-Adesi-Whaley)."""
        q2 = self._q2_factor(r, convenience_yield, sigma)
        d1 = self._d1(S, K, T, r, sigma)
        
        lhs = (1 - exp(-convenience_yield * T) * norm_cdf(d1)) / q2
        rhs = 1 - 1/q2
        
        return lhs - rhs
    
    def _put_exercise_condition(self, S, K, T, r, sigma, convenience_yield):
        """Condition for put early exercise (Barone-Adesi-Whaley)."""
        try:
            q1 = self._q1_factor(r, convenience_yield, sigma)
            
            # q1 should be negative for puts
            if q1 >= 0:
                return -1  # Signal that early exercise is not optimal
            
            d1 = self._d1(S, K, T, r, sigma)
            
            # Avoid numerical issues
            exp_term = exp(-convenience_yield * T)
            if exp_term > 1e10:  # Overflow protection
                return -1
                
            norm_term = norm_cdf(-d1)
            
            lhs = (1 - exp_term * norm_term) / q1
            rhs = 1 - 1/q1
            
            return lhs - rhs
            
        except (ValueError, ZeroDivisionError, OverflowError):
            return -1  # Signal that early exercise is not optimal

# -----------------------------------------------------------------------------
# GREEKS CALCULATION

class AmericanOptionGreeks:
    """
    Calculate Greeks for American options using finite difference methods.
    """
    
    def __init__(self, pricer=None):
        self.pricer = pricer or AmericanBlack76Pricer()
        
        # Finite difference parameters
        self.delta_S = 0.01      # 1% bump for delta
        self.delta_vol = 0.0001  # 0.01% bump for vega  
        self.delta_r = 0.0001    # 0.01% bump for rho
        self.delta_t = 1/365     # 1 day for theta
        self.delta_cy = 0.0001   # 0.01% bump for convenience yield sensitivity
    
    def delta(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Delta: sensitivity to underlying price changes.
        ∂V/∂F
        """
        # Central difference
        F_up = F * (1 + self.delta_S)
        F_down = F * (1 - self.delta_S)
        
        price_up = self.pricer.american_option_price(F_up, K, T, r, sigma, option_type, method, convenience_yield)
        price_down = self.pricer.american_option_price(F_down, K, T, r, sigma, option_type, method, convenience_yield)
        
        return (price_up - price_down) / (F_up - F_down)
    
    def gamma(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Gamma: second derivative with respect to underlying price.
        ∂²V/∂F²
        """
        # Central difference for second derivative
        F_up = F * (1 + self.delta_S)
        F_down = F * (1 - self.delta_S)
        
        price_center = self.pricer.american_option_price(F, K, T, r, sigma, option_type, method, convenience_yield)
        price_up = self.pricer.american_option_price(F_up, K, T, r, sigma, option_type, method, convenience_yield)
        price_down = self.pricer.american_option_price(F_down, K, T, r, sigma, option_type, method, convenience_yield)
        
        dF = F * self.delta_S
        return (price_up - 2*price_center + price_down) / (dF**2)
    
    def vega(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Vega: sensitivity to volatility changes.
        ∂V/∂σ
        """
        sigma_up = sigma + self.delta_vol
        sigma_down = sigma - self.delta_vol
        
        price_up = self.pricer.american_option_price(F, K, T, r, sigma_up, option_type, method, convenience_yield)
        price_down = self.pricer.american_option_price(F, K, T, r, sigma_down, option_type, method, convenience_yield)
        
        return (price_up - price_down) / (2 * self.delta_vol)
    
    def theta(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Theta: time decay.
        ∂V/∂T (usually reported as -∂V/∂T to show decay)
        """
        if T <= self.delta_t:
            # Near expiration, use smaller time bump
            dt = min(self.delta_t, T/2)
        else:
            dt = self.delta_t
        
        T_down = T - dt
        
        price_current = self.pricer.american_option_price(F, K, T, r, sigma, option_type, method, convenience_yield)
        
        if T_down <= 0:
            # At expiration, return intrinsic value
            if option_type == 'call':
                price_expiry = max(F - K, 0)
            else:
                price_expiry = max(K - F, 0)
        else:
            price_expiry = self.pricer.american_option_price(F, K, T_down, r, sigma, option_type, method, convenience_yield)
        
        # Return negative of time derivative (conventional theta)
        return -(price_current - price_expiry) / dt
    
    def rho(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Rho: sensitivity to interest rate changes.
        ∂V/∂r
        """
        r_up = r + self.delta_r
        r_down = r - self.delta_r
        
        price_up = self.pricer.american_option_price(F, K, T, r_up, sigma, option_type, method, convenience_yield)
        price_down = self.pricer.american_option_price(F, K, T, r_down, sigma, option_type, method, convenience_yield)
        
        return (price_up - price_down) / (2 * self.delta_r)
    
    def cy_sensitivity(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Convenience Yield Sensitivity: sensitivity to convenience yield changes.
        ∂V/∂cy (specific to commodity options)
        """
        cy_up = convenience_yield + self.delta_cy
        cy_down = max(0, convenience_yield - self.delta_cy)  # Don't go negative
        
        price_up = self.pricer.american_option_price(F, K, T, r, sigma, option_type, method, cy_up)
        price_down = self.pricer.american_option_price(F, K, T, r, sigma, option_type, method, cy_down)
        
        return (price_up - price_down) / (cy_up - cy_down)
    
    def all_greeks(self, F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
        """
        Calculate all Greeks at once for efficiency.
        
        Returns:
        --------
        dict : All Greeks values
        """
        return {
            'price': self.pricer.american_option_price(F, K, T, r, sigma, option_type, method, convenience_yield),
            'delta': self.delta(F, K, T, r, sigma, option_type, convenience_yield, method),
            'gamma': self.gamma(F, K, T, r, sigma, option_type, convenience_yield, method),
            'vega': self.vega(F, K, T, r, sigma, option_type, convenience_yield, method),
            'theta': self.theta(F, K, T, r, sigma, option_type, convenience_yield, method),
            'rho': self.rho(F, K, T, r, sigma, option_type, convenience_yield, method),
            'cy_sensitivity': self.cy_sensitivity(F, K, T, r, sigma, option_type, convenience_yield, method)
        }

# -----------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS FOR GREEKS

def american_option_greeks(F, K, T, r, sigma, option_type='call', convenience_yield=0.0, method='barone_adesi_whaley'):
    """
    Calculate all Greeks for an American option.
    
    Parameters:
    -----------
    F : float
        Current futures price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
    convenience_yield : float
        Convenience yield
    method : str
        'barone_adesi_whaley' or 'bjerksund_stensland'
        
    Returns:
    --------
    dict : All Greeks and option price
    """
    greeks_calc = AmericanOptionGreeks()
    return greeks_calc.all_greeks(F, K, T, r, sigma, option_type, convenience_yield, method)

# -----------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS

def american_call(F, K, T, r, sigma, convenience_yield=0.0, method='barone_adesi_whaley'):
    """
    American call option price on futures.
    
    Parameters:
    -----------
    F : float
        Current futures price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free rate
    sigma : float
        Volatility
    convenience_yield : float
        Convenience yield of commodity
    method : str
        'barone_adesi_whaley' or 'bjerksund_stensland'
    
    Returns:
    --------
    float : American call option price
    """
    pricer = AmericanBlack76Pricer()
    return pricer.american_option_price(F, K, T, r, sigma, 'call', method, convenience_yield)

def american_put(F, K, T, r, sigma, convenience_yield=0.0, method='barone_adesi_whaley'):
    """
    American put option price on futures.
    
    Parameters:
    -----------
    F : float
        Current futures price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free rate
    sigma : float
        Volatility
    convenience_yield : float
        Convenience yield of commodity
    method : str
        'barone_adesi_whaley' or 'bjerksund_stensland'
    
    Returns:
    --------
    float : American put option price
    """
    pricer = AmericanBlack76Pricer()
    return pricer.american_option_price(F, K, T, r, sigma, 'put', method, convenience_yield)

def american_option(option_type, F, K, T, r, sigma, convenience_yield=0.0, method='barone_adesi_whaley'):
    """
    American option price with string option type.
    
    Parameters:
    -----------
    option_type : str
        'call' or 'put'
    """
    pricer = AmericanBlack76Pricer()
    return pricer.american_option_price(F, K, T, r, sigma, option_type, method, convenience_yield)

# -----------------------------------------------------------------------------
# TESTING

def _test_american_options():
    """
    Test function following py_lets_be_rational pattern.
    
    >>> F = 100.0
    >>> K = 105.0
    >>> T = 0.25
    >>> r = 0.05
    >>> sigma = 0.20
    >>> convenience_yield = 0.02
    
    >>> call_price = american_call(F, K, T, r, sigma, convenience_yield)
    >>> put_price = american_put(F, K, T, r, sigma, convenience_yield)
    
    >>> call_price > 0
    True
    >>> put_price > 0
    True
    
    >>> # American put should be worth more than European put (for ITM case)
    >>> pricer = AmericanBlack76Pricer()
    >>> european_put = pricer.black76_european(80, 100, T, r, sigma, 'put')
    >>> american_put_price = american_put(80, 100, T, r, sigma, convenience_yield)
    >>> american_put_price >= european_put
    True
    """
    pass

if __name__ == '__main__':
    import doctest
    if not doctest.testmod().failed:
        print("American options module: All tests passed.")