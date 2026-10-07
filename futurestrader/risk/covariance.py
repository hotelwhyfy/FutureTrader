"""Covariance estimation for the risk layer.

Why this file exists at all: a 43x43 sample covariance estimated from a few
years of daily data is badly conditioned. The smallest eigenvalues are mostly
noise, and any optimiser or risk decomposition that inverts or leans on them
will produce confident nonsense. Two standard corrections are applied:

  EWMA weighting   recent observations count more, so the estimate tracks
                   regime change instead of averaging 2008 with today.
  Shrinkage        pull the correlation matrix toward its own mean off-diagonal,
                   which stabilises the small eigenvalues at the cost of a
                   little bias. Ledoit-Wolf's argument, applied to correlations
                   rather than covariances so that volatilities stay untouched.

Unlike expected returns, these quantities are genuinely estimable: split-sample
correlation persistence across this universe is 0.84 and volatility persistence
0.88, against 0.50 for Sharpe. That asymmetry is the whole reason the risk layer
is worth more than the signal layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_cov(returns: pd.DataFrame, halflife: int = 126, min_periods: int = 60) -> pd.DataFrame:
    """Exponentially weighted covariance, annualised.

    `halflife` is in trading days: 126 ~ six months, the RiskMetrics-ish default.
    Shorter reacts faster and is noisier.
    """
    r = returns.dropna(how="all")
    if len(r) < min_periods:
        raise ValueError(f"need >= {min_periods} observations, got {len(r)}")

    w = 0.5 ** (np.arange(len(r))[::-1] / halflife)
    w = w / w.sum()
    x = r.to_numpy(dtype=float)
    # Missing bars are filled with the weighted mean so they contribute no
    # spurious covariance; instruments with short histories are handled by
    # `align` below rather than being silently treated as zero-return.
    mu = np.nansum(x * w[:, None], axis=0)
    xc = np.where(np.isnan(x), 0.0, x - mu)
    cov = (xc * w[:, None]).T @ xc
    cov *= TRADING_DAYS
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def shrink_correlation(cov: pd.DataFrame, intensity: float = 0.2) -> pd.DataFrame:
    """Shrink correlations toward the average pairwise correlation.

    intensity 0 leaves the sample estimate alone; 1 replaces every correlation
    with the universe average. 0.1-0.3 is the useful range.
    """
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must be in [0, 1]")
    sd = np.sqrt(np.diag(cov.to_numpy()))
    sd_safe = np.where(sd > 0, sd, np.nan)
    corr = cov.to_numpy() / np.outer(sd_safe, sd_safe)
    np.fill_diagonal(corr, 1.0)

    n = corr.shape[0]
    if n > 1:
        off = corr[~np.eye(n, dtype=bool)]
        target = np.full_like(corr, np.nanmean(off))
        np.fill_diagonal(target, 1.0)
        corr = (1 - intensity) * corr + intensity * target
        np.fill_diagonal(corr, 1.0)

    out = corr * np.outer(sd, sd)
    return pd.DataFrame(out, index=cov.index, columns=cov.columns)


def estimate(
    returns: pd.DataFrame,
    halflife: int = 126,
    shrinkage: float = 0.2,
    lookback_days: int | None = 756,
) -> pd.DataFrame:
    """The estimator the risk layer uses: EWMA plus correlation shrinkage."""
    r = returns.tail(lookback_days) if lookback_days else returns
    return shrink_correlation(ewma_cov(r, halflife=halflife), shrinkage)


def align(returns: dict[str, pd.Series], symbols: list[str], min_overlap: int = 60):
    """Build a return matrix over symbols, dropping those with too little history.

    Returns (frame, dropped). Short-history instruments are reported rather than
    silently zero-filled, because a zero-return column looks like a risk-free
    asset and would understate portfolio risk.
    """
    cols, dropped = {}, []
    for s in symbols:
        ser = returns.get(s)
        if ser is None or ser.dropna().shape[0] < min_overlap:
            dropped.append(s)
            continue
        cols[s] = ser
    if not cols:
        raise ValueError("no instrument had enough history to estimate risk")
    frame = pd.DataFrame(cols).sort_index()
    # Keep the span where at least half the book is observable.
    frame = frame[frame.notna().sum(axis=1) >= max(1, frame.shape[1] // 2)]
    return frame, dropped
