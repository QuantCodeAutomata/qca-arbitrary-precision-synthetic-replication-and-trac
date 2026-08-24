"""Black-Scholes European option pricing and Greeks using arbitrary-precision arithmetic.

All calculations use decimal.Decimal (getcontext().prec = 28).
Standard float64 / double types are strictly forbidden per specification.

# Custom — Context7 found no library equivalent for Decimal-native BS Greeks
"""
from decimal import Decimal, getcontext
from typing import Literal

from src.precision import D, ONE, TWO, ZERO, to_decimal, quantize8

getcontext().prec = 28

_PI = D("3.14159265358979323846")


def _sqrt(x: Decimal) -> Decimal:
    """Compute square root in Decimal arithmetic (built-in method)."""
    if x < ZERO:
        raise ValueError(f"Cannot take sqrt of negative: {x}")
    if x == ZERO:
        return ZERO
    return x.sqrt()


def _exp(x: Decimal) -> Decimal:
    """Compute e^x in Decimal arithmetic (built-in method)."""
    return x.exp()


def _ln(x: Decimal) -> Decimal:
    """Compute natural logarithm in Decimal arithmetic (built-in method)."""
    if x <= ZERO:
        raise ValueError(f"ln undefined for x <= 0: {x}")
    return x.ln()


def _norm_cdf(x: Decimal) -> Decimal:
    """Standard normal CDF using Abramowitz & Stegun approximation in Decimal.

    # Custom — Context7 found no library equivalent for Decimal-native CDF
    Coefficients from A&S formula 26.2.17 (maximum |error| < 7.5e-8).
    """
    a1 = D("0.31938153")
    a2 = D("-0.356563782")
    a3 = D("1.781477937")
    a4 = D("-1.821255978")
    a5 = D("1.330274429")
    p = D("0.2316419")

    x_abs = abs(x)
    t = ONE / (ONE + p * x_abs)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    pdf_val = _exp(D("-0.5") * x_abs * x_abs) / _sqrt(TWO * _PI)
    cdf_pos = ONE - pdf_val * poly

    if x >= ZERO:
        return cdf_pos
    return ONE - cdf_pos


def _norm_pdf(x: Decimal) -> Decimal:
    """Standard normal PDF in Decimal arithmetic."""
    return _exp(D("-0.5") * x * x) / _sqrt(TWO * _PI)


def bs_d1(S: Decimal, K: Decimal, r: Decimal, sigma: Decimal, T: Decimal) -> Decimal:
    """Compute d1 in the Black-Scholes formula.

    Parameters
    ----------
    S     : Current underlying price.
    K     : Strike price.
    r     : Risk-free rate (annualised, continuously compounded).
    sigma : Implied volatility (annualised).
    T     : Time to expiration in years.
    """
    if T <= ZERO:
        raise ValueError("Time to expiration T must be positive.")
    return (_ln(S / K) + (r + sigma ** TWO / TWO) * T) / (sigma * _sqrt(T))


def bs_d2(d1: Decimal, sigma: Decimal, T: Decimal) -> Decimal:
    """Compute d2 = d1 - sigma * sqrt(T)."""
    return d1 - sigma * _sqrt(T)


def bs_call_price(
    S: Decimal, K: Decimal, r: Decimal, sigma: Decimal, T: Decimal
) -> Decimal:
    """Black-Scholes European call price.

    Parameters
    ----------
    S, K, r, sigma, T : Standard Black-Scholes parameters.
    """
    S, K, r, sigma, T = map(to_decimal, [S, K, r, sigma, T])
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    return S * _norm_cdf(d1) - K * _exp(-r * T) * _norm_cdf(d2)


def bs_put_price(
    S: Decimal, K: Decimal, r: Decimal, sigma: Decimal, T: Decimal
) -> Decimal:
    """Black-Scholes European put price.

    Parameters
    ----------
    S, K, r, sigma, T : Standard Black-Scholes parameters.
    """
    S, K, r, sigma, T = map(to_decimal, [S, K, r, sigma, T])
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    return K * _exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def delta(
    S: Decimal,
    K: Decimal,
    r: Decimal,
    sigma: Decimal,
    T: Decimal,
    option_type: Literal["call", "put"] = "call",
) -> Decimal:
    """Black-Scholes Delta (dV/dS).

    Parameters
    ----------
    S, K, r, sigma, T : Standard BS parameters.
    option_type       : 'call' or 'put'.
    """
    S, K, r, sigma, T = map(to_decimal, [S, K, r, sigma, T])
    if T <= ZERO:
        return ZERO
    d1 = bs_d1(S, K, r, sigma, T)
    if option_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - ONE


def gamma(
    S: Decimal, K: Decimal, r: Decimal, sigma: Decimal, T: Decimal
) -> Decimal:
    """Black-Scholes Gamma (d^2V/dS^2) – identical for call and put.

    Parameters
    ----------
    S, K, r, sigma, T : Standard BS parameters.
    """
    S, K, r, sigma, T = map(to_decimal, [S, K, r, sigma, T])
    if T <= ZERO:
        return ZERO
    d1 = bs_d1(S, K, r, sigma, T)
    return _norm_pdf(d1) / (S * sigma * _sqrt(T))


def theta(
    S: Decimal,
    K: Decimal,
    r: Decimal,
    sigma: Decimal,
    T: Decimal,
    option_type: Literal["call", "put"] = "call",
) -> Decimal:
    """Black-Scholes Theta (dV/dt) per calendar day.

    Parameters
    ----------
    S, K, r, sigma, T : Standard BS parameters.
    option_type       : 'call' or 'put'.
    """
    S, K, r, sigma, T = map(to_decimal, [S, K, r, sigma, T])
    if T <= ZERO:
        return ZERO
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    term1 = -(S * _norm_pdf(d1) * sigma) / (TWO * _sqrt(T))
    if option_type == "call":
        term2 = r * K * _exp(-r * T) * _norm_cdf(d2)
        return (term1 - term2) / D("365")
    term2 = r * K * _exp(-r * T) * _norm_cdf(-d2)
    return (term1 + term2) / D("365")


def vega(
    S: Decimal, K: Decimal, r: Decimal, sigma: Decimal, T: Decimal
) -> Decimal:
    """Black-Scholes Vega (dV/dsigma) – identical for call and put.

    Parameters
    ----------
    S, K, r, sigma, T : Standard BS parameters.
    """
    S, K, r, sigma, T = map(to_decimal, [S, K, r, sigma, T])
    if T <= ZERO:
        return ZERO
    d1 = bs_d1(S, K, r, sigma, T)
    return S * _norm_pdf(d1) * _sqrt(T)
