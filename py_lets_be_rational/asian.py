# -*- coding: utf-8 -*-
"""
py_lets_be_rational.asian
~~~~~~~~~~~~~~~~~~~~~~~~~

Asian option pricing using the Turnbull-Wakeman approximation.
This module extends py_lets_be_rational to support Asian options with
proper averaging window handling.

:copyright: © 2025 Extended Implementation  
:license: MIT, see LICENSE for more details.

About Turnbull-Wakeman:
~~~~~~~~~~~~~~~~~~~~~~~

Turnbull, S.M. & Wakeman, L.M. (1991). "A Quick Algorithm for 
Pricing European Average Options." Journal of Financial and 
Quantitative Analysis, 26(3), 377-389.

About LetsBeRational:
~~~~~~~~~~~~~~~~~~~~~

The source code of LetsBeRational resides at www.jaeckel.org/LetsBeRational.7z .

======================================================================================
Copyright © 2013-2014 Peter Jäckel.

Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

WARRANTY DISCLAIMER
The Software is provided "as is" without warranty of any kind, either express or implied,
including without limitation any implied warranties of condition, uninterrupted use,
merchantability, fitness for a particular purpose, or non-infringement.
======================================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS

# Standard library imports
from math import log, sqrt, exp
import warnings

# Related third party imports
import numpy as np

# Local application/library specific imports
from py_lets_be_rational import constants
from py_lets_be_rational.exceptions import VolatilityValueException
from py_lets_be_rational.numba_helper import maybe_jit
from py_lets_be_rational.normaldistribution import norm_cdf

# -----------------------------------------------------------------------------
# CONSTANTS

CALL = 1
PUT = -1

# -----------------------------------------------------------------------------
# CORE FUNCTIONS

@maybe_jit(cache=True, nopython=True)
def _calculate_moments_with_window(S, T_option, r, sigma, b, T_start, T_end, 
                                  current_time=0.0, avg_so_far=None, observations_so_far=0):
    """
    Calculate moments for Asian option with explicit averaging window.
    
    Parameters:
    -----------
    S : float
        Current asset price
    T_option : float
        Option expiration time
    r : float
        Risk-free rate
    sigma : float
        Volatility
    b : float
        Cost of carry
    T_start : float
        Time when averaging begins
    T_end : float
        Time when averaging ends
    current_time : float
        Current time (default 0)
    avg_so_far : float, optional
        Average accumulated so far (if averaging has started)
    observations_so_far : int
        Number of observations included in avg_so_far
    
    Returns:
    --------
    tuple : (M1, M2) - First and second moments of the average
    """
    
    # Validate timing
    if T_start < 0 or T_end < T_start or T_option < T_end:
        raise ValueError("Invalid timing: T_start >= 0, T_end >= T_start, T_option >= T_end")
    
    averaging_length = T_end - T_start
    
    if averaging_length <= constants.SQRT_DBL_EPSILON:
        # Degenerate case: no averaging period
        return S, S * S
    
    # Case 1: Averaging hasn't started yet
    if current_time <= T_start:
        return _moments_pre_averaging(S, T_start, T_end, r, sigma, b, current_time)
    
    # Case 2: Averaging has ended
    elif current_time >= T_end:
        if avg_so_far is None:
            raise ValueError("avg_so_far must be provided when averaging has ended")
        return avg_so_far, avg_so_far * avg_so_far  # Deterministic
    
    # Case 3: Currently in averaging period
    else:
        return _moments_during_averaging(S, T_start, T_end, r, sigma, b, 
                                       current_time, avg_so_far, observations_so_far)

@maybe_jit(cache=True, nopython=True)
def _moments_pre_averaging(S, T_start, T_end, r, sigma, b, current_time):
    """Calculate moments when averaging hasn't started yet."""
    
    time_to_start = T_start - current_time
    averaging_length = T_end - T_start
    
    # Forward price at averaging start
    F_start = S * exp(b * time_to_start)
    
    # Expected average during the averaging period
    if abs(b) < constants.SQRT_DBL_EPSILON:  # Zero cost of carry
        M1 = F_start
        M2 = F_start * F_start * (1.0 + sigma * sigma * averaging_length / 3.0)
    else:
        # Non-zero cost of carry
        M1 = F_start * (exp(b * averaging_length) - 1.0) / (b * averaging_length)
        
        # Calculate second moment
        sigma_sq = sigma * sigma
        b_plus_sigma = b + sigma_sq / 2.0
        
        if abs(b_plus_sigma) < constants.SQRT_DBL_EPSILON:
            M2 = F_start * F_start * (1.0 + sigma_sq * averaging_length / 3.0)
        else:
            exp_2b_T = exp(2.0 * b * averaging_length)
            exp_b_sigma_T = exp(b_plus_sigma * averaging_length)
            
            T_sq = averaging_length * averaging_length
            term1 = 2.0 * exp_b_sigma_T / (b_plus_sigma * T_sq)
            term2 = 2.0 / (b * T_sq)
            term3 = 2.0 * exp_2b_T / ((2.0 * b + sigma_sq) * T_sq)
            
            factor = (F_start * F_start) / (2.0 * b + sigma_sq)
            M2 = factor * (exp_2b_T - 1.0) * (term1 - term2 + term3)
    
    return M1, M2

@maybe_jit(cache=True, nopython=True)
def _moments_during_averaging(S, T_start, T_end, r, sigma, b, current_time, 
                            avg_so_far, observations_so_far):
    """Calculate moments when currently in averaging period."""
    
    time_elapsed = current_time - T_start
    time_remaining = T_end - current_time
    averaging_length = T_end - T_start
    
    # Handle case where no averaging has been recorded yet
    if avg_so_far is None:
        # Assume averaging just started, use current price as initial average
        if time_elapsed <= constants.SQRT_DBL_EPSILON:
            avg_so_far = S
            observations_so_far = 1
        else:
            # Need to estimate what the average should be
            # This is a limitation - in practice, we need the actual accumulated average
            avg_so_far = S  # Rough approximation
            observations_so_far = max(1, int(time_elapsed * 252))  # Rough daily observations
    
    if time_remaining <= constants.SQRT_DBL_EPSILON:
        # Averaging just ended
        return avg_so_far, avg_so_far * avg_so_far
    
    # Weight of past vs future averaging
    weight_past = time_elapsed / averaging_length
    weight_future = time_remaining / averaging_length
    
    # Expected average for remaining period
    if abs(b) < constants.SQRT_DBL_EPSILON:  # Zero cost of carry
        E_future = S
        Var_future = (S * S) * (sigma * sigma) * time_remaining / 3.0
    else:
        E_future = S * (exp(b * time_remaining) - 1.0) / (b * time_remaining)
        
        # Variance calculation for remaining period
        sigma_sq = sigma * sigma
        if abs(b - sigma_sq / 2.0) < constants.SQRT_DBL_EPSILON:
            Var_future = (S * S) * sigma_sq * time_remaining / 3.0
        else:
            term1 = 2.0 * exp(b * time_remaining) / ((b + sigma_sq / 2.0) * time_remaining * time_remaining)
            term2 = 2.0 / (b * time_remaining * time_remaining)
            term3 = 2.0 / ((b + sigma_sq / 2.0) * time_remaining)
            Var_future = (S * S / (2.0 * b + sigma_sq)) * (term1 - term2 - term3)
    
    # Combined moments
    M1 = weight_past * avg_so_far + weight_future * E_future
    M2 = M1 * M1 + (weight_future * weight_future) * Var_future
    
    return M1, M2

@maybe_jit(cache=True, nopython=True)
def _turnbull_wakeman_price(M1, M2, K, T, r, flag):
    """Core Turnbull-Wakeman pricing calculation."""
    
    if M2 <= M1 * M1 * (1.0 + constants.SQRT_DBL_EPSILON):
        # Deterministic case
        if flag == CALL:
            return max(M1 - K, 0.0) * exp(-r * T)
        else:
            return max(K - M1, 0.0) * exp(-r * T)
    
    # Log-normal approximation parameters
    sigma_A_squared = log(M2 / (M1 * M1))
    sigma_A = sqrt(sigma_A_squared)
    mu_A = log(M1) - 0.5 * sigma_A_squared
    
    # Black-Scholes style formula with adjusted parameters
    d1 = (mu_A - log(K) + 0.5 * sigma_A_squared) / sigma_A
    d2 = d1 - sigma_A
    
    discount = exp(-r * T)
    exp_mu_sigma = exp(mu_A + 0.5 * sigma_A_squared)
    
    if flag == CALL:
        price = discount * (exp_mu_sigma * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        price = discount * (K * norm_cdf(-d2) - exp_mu_sigma * norm_cdf(-d1))
        
    return price

# -----------------------------------------------------------------------------
# PUBLIC API FUNCTIONS

def asian_option_with_window(flag, S, K, T_option, r, sigma, T_start, T_end,
                           cost_of_carry=None, current_time=0.0, 
                           avg_so_far=None, observations_so_far=0):
    """
    Price Asian option with explicit averaging window.
    
    Parameters:
    -----------
    flag : int
        1 for call, -1 for put
    S : float
        Current asset price
    K : float
        Strike price
    T_option : float
        Option expiration time
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T_start : float
        Time when averaging begins
    T_end : float
        Time when averaging ends
    cost_of_carry : float, optional
        Cost of carry (default: 0 for futures)
    current_time : float
        Current time (default: 0)
    avg_so_far : float, optional
        Average accumulated so far
    observations_so_far : int
        Number of observations in average so far
        
    Returns:
    --------
    float : Option price
    """
    
    if sigma < 0:
        raise VolatilityValueException()
    
    if cost_of_carry is None:
        b = 0.0
    else:
        b = cost_of_carry
    
    # Calculate moments with proper window handling
    M1, M2 = _calculate_moments_with_window(S, T_option, r, sigma, b, T_start, T_end,
                                          current_time, avg_so_far, observations_so_far)
    
    # Price using standard Turnbull-Wakeman formula
    return _turnbull_wakeman_price(M1, M2, K, T_option, r, flag)

def turnbull_wakeman_call(S, K, T, r, sigma, cost_of_carry=None):
    """
    Calculate Asian call option price using Turnbull-Wakeman approximation.
    
    Parameters:
    -----------
    S : float
        Current asset price (spot or futures price)
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of underlying asset
    cost_of_carry : float, optional
        Cost of carry rate. If None, assumes futures (b=0)
        
    Returns:
    --------
    float : Asian call option price
    """
    return asian_option_with_window(CALL, S, K, T, r, sigma, 0.0, T, cost_of_carry)

def turnbull_wakeman_put(S, K, T, r, sigma, cost_of_carry=None):
    """
    Calculate Asian put option price using Turnbull-Wakeman approximation.
    
    Parameters:
    -----------
    S : float
        Current asset price (spot or futures price)
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of underlying asset
    cost_of_carry : float, optional
        Cost of carry rate. If None, assumes futures (b=0)
        
    Returns:
    --------
    float : Asian put option price
    """
    return asian_option_with_window(PUT, S, K, T, r, sigma, 0.0, T, cost_of_carry)

def turnbull_wakeman(flag, S, K, T, r, sigma, cost_of_carry=None):
    """
    Calculate Asian option price using Turnbull-Wakeman approximation.
    
    Parameters:
    -----------
    flag : int
        Option type: 1 for call, -1 for put
    S : float
        Current asset price (spot or futures price)
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of underlying asset
    cost_of_carry : float, optional
        Cost of carry rate. If None, assumes futures (b=0)
        
    Returns:
    --------
    float : Asian option price
    """
    if flag == CALL:
        return turnbull_wakeman_call(S, K, T, r, sigma, cost_of_carry)
    elif flag == PUT:
        return turnbull_wakeman_put(S, K, T, r, sigma, cost_of_carry)
    else:
        raise ValueError("flag must be 1 (call) or -1 (put)")

def turnbull_wakeman_with_partial_averaging(flag, S, K, T, r, sigma, 
                                           avg_price_so_far, time_elapsed,
                                           cost_of_carry=None):
    """
    Calculate Asian option price with partial averaging elapsed.
    
    Parameters:
    -----------
    flag : int
        Option type: 1 for call, -1 for put
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Total time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of underlying asset
    avg_price_so_far : float
        Average price accumulated so far
    time_elapsed : float
        Time already elapsed in averaging period
    cost_of_carry : float, optional
        Cost of carry rate. If None, assumes futures (b=0)
        
    Returns:
    --------
    float : Asian option price
    """
    return asian_option_with_window(flag, S, K, T, r, sigma, 0.0, T, 
                                  cost_of_carry, time_elapsed, avg_price_so_far)

def turnbull_wakeman_vectorized(flag, S, K, T, r, sigma, cost_of_carry=None):
    """
    Vectorized Turnbull-Wakeman pricing function.
    
    All parameters can be scalars or numpy arrays for batch processing.
    """
    # Convert inputs to numpy arrays
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    if cost_of_carry is None:
        b = np.zeros_like(r)
    else:
        b = np.asarray(cost_of_carry, dtype=float)
    
    # Check for invalid volatilities
    if np.any(sigma < 0):
        raise VolatilityValueException()
    
    # Calculate for each element
    results = np.zeros_like(S)
    flat_indices = np.ndindex(S.shape)
    
    for idx in flat_indices:
        S_i = S[idx] if S.shape else S.item()
        K_i = K[idx] if K.shape else K.item()
        T_i = T[idx] if T.shape else T.item()
        r_i = r[idx] if r.shape else r.item()
        sigma_i = sigma[idx] if sigma.shape else sigma.item()
        b_i = b[idx] if b.shape else b.item()
        
        results[idx] = asian_option_with_window(flag, S_i, K_i, T_i, r_i, sigma_i, 
                                             0.0, T_i, b_i)
    
    return results

def turnbull_wakeman_implied_volatility(price, flag, S, K, T, r, cost_of_carry=None):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    -----------
    price : float
        Market price of the Asian option
    flag : int
        Option type: 1 for call, -1 for put
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    cost_of_carry : float, optional
        Cost of carry rate. If None, assumes futures (b=0)
        
    Returns:
    --------
    float : Implied volatility
    """
    
    def objective(vol):
        return turnbull_wakeman(flag, S, K, T, r, vol, cost_of_carry) - price
    
    def vega_calc(vol):
        h = 0.001
        price_up = turnbull_wakeman(flag, S, K, T, r, vol + h, cost_of_carry)
        price_down = turnbull_wakeman(flag, S, K, T, r, vol - h, cost_of_carry)
        return (price_up - price_down) / (2.0 * h)
    
    # Initial guess
    vol = 0.2
    max_iterations = 50
    tolerance = 1e-8
    
    for iteration in range(max_iterations):
        price_calc = turnbull_wakeman(flag, S, K, T, r, vol, cost_of_carry)
        diff = price_calc - price
        
        if abs(diff) < tolerance:
            break
            
        vega = vega_calc(vol)
        
        if abs(vega) < 1e-10:
            break
            
        vol_new = vol - diff / vega
        
        if abs(vol_new - vol) < tolerance:
            break
            
        vol = max(vol_new, 0.001)  # Ensure positive volatility
    
    return vol

# -----------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS FOR COMMON PATTERNS

def asian_last_n_days(flag, S, K, T_option, r, sigma, averaging_days=30,
                     cost_of_carry=None, current_time=0.0, avg_so_far=None):
    """
    Asian option averaged over last N days before expiration.
    
    Common pattern: Average over last 30 days before expiration.
    """
    T_end = T_option
    T_start = max(0, T_option - averaging_days / 365.25)
    
    return asian_option_with_window(flag, S, K, T_option, r, sigma, T_start, T_end,
                                  cost_of_carry, current_time, avg_so_far)

def asian_calendar_period(flag, S, K, T_option, r, sigma, year_start, year_end,
                         cost_of_carry=None, current_time=0.0, avg_so_far=None):
    """
    Asian option averaged over specific calendar period.
    
    Example: Average over calendar year regardless of option expiration.
    """
    T_start = year_start
    T_end = year_end
    
    return asian_option_with_window(flag, S, K, T_option, r, sigma, T_start, T_end,
                                  cost_of_carry, current_time, avg_so_far)

def asian_delayed_start(flag, S, K, T_option, r, sigma, delay_months=6,
                       cost_of_carry=None, current_time=0.0):
    """
    Asian option where averaging starts after a delay.
    
    Example: 1-year option where averaging only starts after 6 months.
    """
    T_start = delay_months / 12.0
    T_end = T_option
    
    return asian_option_with_window(flag, S, K, T_option, r, sigma, T_start, T_end,
                                  cost_of_carry, current_time)

# -----------------------------------------------------------------------------
# COMPATIBILITY LAYER

def asian_option_price(option_type, S, K, T, r, sigma, cost_of_carry=None):
    """
    Convenience function with string option type.
    
    Parameters:
    -----------
    option_type : str
        'call' or 'put'
    """
    flag = CALL if option_type.lower() == 'call' else PUT
    return turnbull_wakeman(flag, S, K, T, r, sigma, cost_of_carry)

# -----------------------------------------------------------------------------
# TESTING AND VALIDATION

def _test_turnbull_wakeman():
    """
    Test function following py_lets_be_rational pattern.
    
    >>> S = 100.0
    >>> K = 105.0
    >>> T = 0.25
    >>> r = 0.05
    >>> sigma = 0.20
    
    >>> call_price = turnbull_wakeman_call(S, K, T, r, sigma)
    >>> put_price = turnbull_wakeman_put(S, K, T, r, sigma)
    
    >>> call_price > 0
    True
    >>> put_price > 0
    True
    
    >>> # Test vectorized
    >>> strikes = [95, 100, 105]
    >>> call_prices = turnbull_wakeman_vectorized(CALL, S, strikes, T, r, sigma)
    >>> len(call_prices) == 3
    True
    """
    pass

if __name__ == '__main__':
    import doctest
    if not doctest.testmod().failed:
        print("Turnbull-Wakeman module: All tests passed.")