# Overview

`py_lets_be_rational` is a pure Python port of `lets_be_rational` extended with Asian option pricing capabilities. Below is a list of differences between the versions:

| Feature                                     | `py_lets_be_rational` v1.1 | `py_lets_be_rational` v1.0 | `lets_be_rational`         |
| ------------------------------------------- |:---------------------------:|:--------------------------:|:--------------------------:|
| Python Version Compatibility                | 2.7 and 3.x                | 2.7 and 3.x               |           2.7 only         |
| Source Language                             | Python                      | Python                     | C with Python SWIG Wrapper |
| Optional Dependencies                       | Numba                      | Numba                      | None                       |
| Asian Option Support                        | ✅ Turnbull-Wakeman        | ❌                         | ❌                         |
| Averaging Window Control                    | ✅ Full control            | ❌                         | ❌                         |
| Vectorized Pricing                          | ✅ NumPy arrays           | ❌                         | ❌                         |
| Installed Automatically by `pip` as part of | py_vollib             | py_vollib                  | vollib                     |

## New in Version 1.1: Asian Option Support

This version adds comprehensive Asian (average) option pricing using the Turnbull-Wakeman approximation:

### Quick Start - Asian Options

```python
import py_lets_be_rational as pylbr

# Basic Asian call option (full averaging period)
asian_call = pylbr.turnbull_wakeman_call(
    S=100,      # Current price
    K=105,      # Strike price  
    T=0.25,     # Time to expiry (3 months)
    r=0.05,     # Risk-free rate
    sigma=0.20  # Volatility
)

# Asian option with custom averaging window
# (e.g., average over last 30 days before expiry)
asian_last_month = pylbr.asian_last_n_days(
    flag=1,              # 1 for call, -1 for put
    S=100, K=105, 
    T_option=1.0,        # 1 year option
    r=0.05, sigma=0.20,
    averaging_days=30    # Average over last 30 days
)

# Vectorized pricing for multiple strikes
strikes = [95, 100, 105, 110]
asian_prices = pylbr.turnbull_wakeman_vectorized(
    flag=1, S=100, K=strikes, T=0.25, r=0.05, sigma=0.20
)
```

### Asian Option Features

- **Flexible Averaging Windows**: Control exactly when averaging starts and ends
- **Partial Averaging Support**: Price options where averaging period has already begun
- **Common Patterns**: Built-in functions for typical Asian option structures
- **High Performance**: Optional Numba acceleration and vectorized operations
- **Zero Cost-of-Carry**: Optimized for futures and commodity options

## Execution Speed
Except for their source language, `py_lets_be_rational` and `lets_be_rational` are almost identical. Each is orders of 
magnitude faster than traditional implied volatility calculation libraries, thanks to the algorithms developed by 
Peter Jaeckel. However, `py_lets_be_rational`, without Numba installed, is about an order of magnitude slower than 
`lets_be_rational`. Numba helps to mitigate this speed gap considerably.

## Dependencies

### Required
- `numpy`: For array operations and mathematical functions

### Optional  
- `numba`: For just-in-time compilation and performance acceleration

## Installing Dependencies

### Basic Installation
```bash
pip install py_lets_be_rational
```

### With Performance Optimization
```bash
pip install py_lets_be_rational[fast]
```

### Manual Numba Installation
`py_lets_be_rational` optionally depends on `numba` which in turn depends on `llvm-lite`. `llvm-lite` wants LLVM 3.9 
being installed. On Mac OSX, use the latest version of HomeBrew to install `numba`'s dependencies as shown below:

```bash
brew install llvm@3.9
LLVM_CONFIG=/usr/local/opt/llvm@3.9/bin/llvm-config pip install llvmlite==0.16.0
pip install numba==0.31.0
```

For other operating systems, please refer to the `llvm-lite` and `numba` documentation.

## Usage Examples

### Black-76 Options (Existing Functionality)
```python
import py_lets_be_rational as pylbr

# European option on futures
european_price = pylbr.black(
    F=100,      # Futures price
    K=105,      # Strike  
    sigma=0.20, # Volatility
    T=0.25,     # Time to expiry
    q=1         # 1 for call, -1 for put  
)

# Implied volatility
iv = pylbr.implied_volatility_from_a_transformed_rational_guess(
    price=5.85, F=100, K=105, T=0.25, q=1
)
```

### Asian Options (New Functionality)

#### Standard Asian Options
```python
# Full averaging period (traditional Asian option)
asian_call = pylbr.turnbull_wakeman_call(100, 105, 0.25, 0.05, 0.20)
asian_put = pylbr.turnbull_wakeman_put(100, 105, 0.25, 0.05, 0.20)

# Using flag notation  
asian_call = pylbr.turnbull_wakeman(1, 100, 105, 0.25, 0.05, 0.20)  # flag=1 for call
asian_put = pylbr.turnbull_wakeman(-1, 100, 105, 0.25, 0.05, 0.20)  # flag=-1 for put
```

#### Real-World Averaging Patterns
```python
# Average over last 3 months before expiration
price = pylbr.asian_last_n_days(
    flag=1, S=100, K=105, T_option=1.0, r=0.05, sigma=0.20,
    averaging_days=90
)

# Average over specific calendar period  
price = pylbr.asian_calendar_period(
    flag=1, S=100, K=105, T_option=1.0, r=0.05, sigma=0.20,
    year_start=0.0, year_end=1.0
)

# Delayed averaging start (6 months into 1-year option)
price = pylbr.asian_delayed_start(
    flag=1, S=100, K=105, T_option=1.0, r=0.05, sigma=0.20,
    delay_months=6
)
```

#### Partial Averaging (Mid-Life Pricing)
```python
# Option where averaging has already started
price = pylbr.asian_option_with_window(
    flag=1,
    S=102,               # Current price
    K=105,               # Strike
    T_option=1.0,        # Total option life
    r=0.05, sigma=0.20,
    T_start=0.25,        # Averaging started 3 months ago  
    T_end=0.75,          # Averaging ends in 3 months
    current_time=0.5,    # Currently 6 months into option
    avg_so_far=98.5      # Average price accumulated so far
)
```

#### Vectorized Batch Pricing
```python
import numpy as np

# Price multiple strikes simultaneously
strikes = np.array([95, 100, 105, 110, 115])
prices = pylbr.turnbull_wakeman_vectorized(
    flag=1, S=100, K=strikes, T=0.25, r=0.05, sigma=0.20
)

# Price with different volatilities
volatilities = np.array([0.15, 0.20, 0.25, 0.30])
prices = pylbr.turnbull_wakeman_vectorized(
    flag=1, S=100, K=105, T=0.25, r=0.05, sigma=volatilities
)
```

#### Implied Volatility for Asian Options
```python
# Calculate Asian option implied volatility
market_price = 3.25
iv = pylbr.turnbull_wakeman_implied_volatility(
    price=market_price, flag=1, S=100, K=105, T=0.25, r=0.05
)
```

## Development

Fork our repository, and make the changes on your local copy:

```bash
git clone git@github.com/your_username/py_lets_be_rational
cd py_lets_be_rational
pip install -e .
pip install -r dev-requirements.txt
```

When you are done, push the changes, and create a pull request on your repo.

## Testing

Run the test suite:

```bash
python -m pytest tests/
python -m doctest py_lets_be_rational/asian.py
```

## About "Let's be Rational"

["Let's Be Rational"](http://www.pjaeckel.webspace.virginmedia.com/LetsBeRational.pdf>) is a paper by [Peter Jäckel](http://jaeckel.org) showing *"how Black's volatility can be implied from option prices with as little as two iterations to maximum attainable precision on standard (64 bit floating point) hardware for all possible inputs."*

The paper is accompanied by the full C source code, which resides at [www.jaeckel.org/LetsBeRational.7z](www.jaeckel.org/LetsBeRational.7z).

## About Turnbull-Wakeman

The Asian option functionality is based on:

**Turnbull, S.M. & Wakeman, L.M. (1991)** - *"A Quick Algorithm for Pricing European Average Options."* Journal of Financial and Quantitative Analysis, 26(3), 377-389.

This implementation includes the modified version for zero cost-of-carry (futures) as used by the European Energy Exchange (EEX) for freight and commodity options.

## License

```
    Copyright © 2013-2014 Peter Jäckel.

    Permission to use, copy, modify, and distribute this software is freely granted,
    provided that this notice is preserved.

    WARRANTY DISCLAIMER
    The Software is provided "as is" without warranty of any kind, either express or implied,
    including without limitation any implied warranties of condition, uninterrupted use,
    merchantability, fitness for a particular purpose, or non-infringement.
```