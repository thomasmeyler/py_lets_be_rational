# -*- coding: utf-8 -*-

"""
py_lets_be_rational.barone_adesi_whaley
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure python implementation extending LetsBeRational to include Barone-Adesi-Whaley approximation for American options.

:copyright: © 2017 Gammon Capital LLC (original LetsBeRational), extended in 2025.
:license: MIT, see LICENSE for more details.

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

from __future__ import division

from math import log, sqrt, exp, fabs

from py_lets_be_rational.numba_helper import maybe_jit
from py_lets_be_rational.lets_be_rational import black, normalised_black
from py_lets_be_rational.normaldistribution import norm_cdf, norm_pdf
from py_lets_be_rational.constants import DBL_EPSILON

@maybe_jit(cache=True, nopython=True)
def black_scholes(S, K, T, r, sigma, b, q):
    """
    General Black-Scholes formula for European option price.

    :param S: Spot price
    :type S: float
    :param K: Strike price
    :type K: float
    :param T: Time to maturity
    :type T: float
    :param r: Risk-free rate
    :type r: float
    :param sigma: Volatility
    :type sigma: float
    :param b: Cost of carry (b = r - d for stocks with dividend yield d; b = 0 for futures; b = r for non-dividend stocks)
    :type b: float
    :param q: +1 for call, -1 for put
    :type q: float
    :return: European option price
    :rtype: float
    """
    if T <= 0:
        return max(q * (S - K), 0.0)
    F = S * exp(b * T)
    discount = exp(-r * T)
    # Use the existing black function for the forward value, then discount
    forward_value = black(F, K, sigma, T, q)
    return discount * forward_value

@maybe_jit(cache=True, nopython=True)
def barone_adesi_whaley(S, K, T, r, sigma, b, q):
    """
    Barone-Adesi-Whaley approximation for American option price.

    :param S: Spot price
    :type S: float
    :param K: Strike price
    :type K: float
    :param T: Time to maturity
    :type T: float
    :param r: Risk-free rate
    :type r: float
    :param sigma: Volatility
    :type sigma: float
    :param b: Cost of carry
    :type b: float
    :param q: +1 for call, -1 for put
    :type q: float
    :return: American option price
    :rtype: float
    """
    if T <= 0:
        return max(q * (S - K), 0.0)

    european = black_scholes(S, K, T, r, sigma, b, q)

    if sigma <= 0:
        return max(european, max(q * (S - K), 0.0))

    h = 1 - exp(-r * T)
    M = 2 * r / (sigma * sigma)
    N_ = 2 * b / (sigma * sigma)

    tolerance = DBL_EPSILON * K * 100  # Reasonable tolerance

    if q == 1:  # American Call
        if b >= r:  # No early exercise
            return european

        q_gamma = 0.5 * (- (N_ - 1) + sqrt((N_ - 1) * (N_ - 1) + 4 * M / h))  # q2

        S_inf = K / (1 - 1 / q_gamma)
        h2 = - (b * T + 2 * sigma * sqrt(T)) * (K / (S_inf - K))
        S_star = K + (S_inf - K) * (1 - exp(h2))

        # Newton iteration
        iterations = 0
        max_iterations = 100
        while iterations < max_iterations:
            iterations += 1
            d1 = (log(S_star / K) + (b + sigma * sigma / 2) * T) / (sigma * sqrt(T))
            n_d1 = norm_cdf(d1)
            lhs = S_star - K
            rhs = black_scholes(S_star, K, T, r, sigma, b, 1) + (1 - exp((b - r) * T) * n_d1) * S_star / q_gamma
            bi = exp((b - r) * T) * n_d1 * (1 - 1 / q_gamma) + (1 - exp((b - r) * T) * norm_pdf(d1) / (sigma * sqrt(T))) / q_gamma
            if fabs(lhs - rhs) < tolerance:
                break
            S_star = S_star - (lhs - rhs) / bi
            if S_star < K:
                S_star = K  # Prevent going below K

        # Now compute A
        d1_star = (log(S_star / K) + (b + sigma * sigma / 2) * T) / (sigma * sqrt(T))
        n_d1_star = norm_cdf(d1_star)
        A = (1 - exp((b - r) * T) * n_d1_star) * S_star / q_gamma

        if S >= S_star:
            return S - K
        else:
            d1 = (log(S / K) + (b + sigma * sigma / 2) * T) / (sigma * sqrt(T))
            n_d1 = norm_cdf(d1)
            eep = A * (S / S_star) ** q_gamma
            return european + eep

    else:  # American Put
        q_gamma = 0.5 * (- (N_ - 1) - sqrt((N_ - 1) * (N_ - 1) + 4 * M / h))  # q1

        S_inf = K / (1 - 1 / q_gamma)
        h1 = (b * T - 2 * sigma * sqrt(T)) * (K / (K - S_inf))  # Note sign change
        S_star = S_inf + (K - S_inf) * exp(h1)  # Note no (1 - )

        # Newton iteration for put
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            d1 = (log(S_star / K) + (b + sigma * sigma / 2) * T) / (sigma * sqrt(T))
            n_minus_d1 = norm_cdf(-d1)
            lhs = K - S_star
            rhs = black_scholes(S_star, K, T, r, sigma, b, -1) + (1 - exp((b - r) * T) * n_minus_d1) * S_star / q_gamma
            bi = - exp((b - r) * T) * n_minus_d1 * (1 - 1 / q_gamma) - (1 + exp((b - r) * T) * norm_pdf(-d1) / (sigma * sqrt(T))) / q_gamma
            if fabs(lhs - rhs) < tolerance:
                break
            S_star = S_star - (lhs - rhs) / bi
            if S_star > K:
                S_star = K  # Prevent going above K for put

        # Compute A for put
        d1_star = (log(S_star / K) + (b + sigma * sigma / 2) * T) / (sigma * sqrt(T))
        n_minus_d1_star = norm_cdf(-d1_star)
        A = (1 - exp((b - r) * T) * n_minus_d1_star) * S_star / q_gamma

        if S <= S_star:
            return K - S
        else:
            d1 = (log(S / K) + (b + sigma * sigma / 2) * T) / (sigma * sqrt(T))
            n_minus_d1 = norm_cdf(-d1)
            eep = A * (S / S_star) ** q_gamma
            return european + eep