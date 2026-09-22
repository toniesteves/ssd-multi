import numpy as np
import pandas as pd


def compute_ssd_dominance(r_portfolio: np.ndarray, r_benchmark: np.ndarray):
    """
    Compute FSD and SSD dominance values between a portfolio and a benchmark.

    Both return series are sorted independently in ascending order (worst to best
    scenario), then:
      FSD_s = r_p_(s) - r_b_(s)                     pointwise difference
      SSD_s = sum_{z=1}^{s} (r_p_(z) - r_b_(z))    cumulative difference

    SSD ≥ 0 everywhere  ⟺  portfolio second-order stochastically dominates benchmark.
    FSD ≥ 0 everywhere  ⟺  portfolio first-order stochastically dominates benchmark.

    Parameters
    ----------
    r_portfolio  : (S,) in-sample portfolio returns
    r_benchmark  : (S,) in-sample benchmark returns (same length)

    Returns
    -------
    s_indices : np.ndarray  shape (S,)  — scenario rank 1..S
    fsd       : np.ndarray  shape (S,)  — FSD relation values
    ssd       : np.ndarray  shape (S,)  — SSD relation values
    """
    r_p  = np.sort(r_portfolio)
    r_b  = np.sort(r_benchmark)
    diff = r_p - r_b
    fsd  = diff
    ssd  = np.cumsum(diff)
    s    = np.arange(1, len(r_p) + 1)
    return s, fsd, ssd


def compute_metrics(series: pd.Series, benchmark: pd.Series = None,
                    rf_annual: float = 0.05) -> dict:
    """
    Compute standard portfolio performance metrics from a normalised price series.

    Parameters
    ----------
    series    : pd.Series — normalised price series (starts at 1.0)
    benchmark : pd.Series — normalised benchmark price series (required for beta)
    rf_annual : float     — annual risk-free rate (default 5%)

    Returns
    -------
    dict with keys:
        Final value, CAGR (%), Mean return (%), Volatility (%), Semi-volatility (%),
        Daily CVaR 5% (%), Sharpe Ratio, Sortino Ratio, Calmar Ratio,
        Max Drawdown (%), Portfolio beta

    Sharpe and Sortino share the same numerator — the annualised excess mean return —
    so the two only differ in the risk measure at the denominator (full volatility vs
    downside semideviation).
    """
    ret = series.pct_change().dropna()
    T   = len(ret)

    final_value = float(series.iloc[-1])
    cagr        = (series.iloc[-1] / series.iloc[0]) ** (252 / T) - 1
    volatility  = ret.std() * np.sqrt(252)

    # Annualised mean return — numerator shared by Sharpe and Sortino
    mean_return = ret.mean() * 252

    # Semi-volatility (downside deviation): sqrt(mean(min(r - target, 0)^2)),
    # target = 0. Squared shortfalls are averaged over all scenarios, not only the
    # negative ones, so it is directly comparable to the full volatility.
    downside_dev = np.sqrt((np.minimum(ret - 0.0, 0.0) ** 2).mean()) * np.sqrt(252)

    # Daily CVaR 5%: average of the worst 5% daily returns (expressed as positive loss)
    var_5pct  = float(ret.quantile(0.05))
    cvar_5pct = float(ret[ret <= var_5pct].mean())

    excess = mean_return - rf_annual

    # Sharpe Ratio (annualised)
    sharpe = excess / volatility if volatility > 0 else np.nan

    # Sortino Ratio: same numerator as Sharpe, downside semideviation at the denominator
    sortino = excess / downside_dev if downside_dev > 0 else np.nan

    # Max Drawdown
    roll_max = series.cummax()
    max_dd   = float(((series - roll_max) / roll_max).min())

    # Calmar Ratio: CAGR per unit of maximum drawdown
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # Portfolio Beta vs benchmark
    beta = np.nan
    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        common    = ret.index.intersection(bench_ret.index)
        if len(common) > 1:
            p = ret.loc[common].values
            b = bench_ret.loc[common].values
            beta = float(np.cov(p, b)[0, 1] / np.var(b, ddof=1))

    return {
        'Final value':         round(final_value, 4),
        'CAGR (%)':            round(cagr * 100, 2),
        'Mean return (%)':     round(mean_return * 100, 2),
        'Volatility (%)':      round(volatility * 100, 2),
        'Semi-volatility (%)': round(downside_dev * 100, 2),
        'Daily CVaR 5% (%)':   round(abs(cvar_5pct) * 100, 3),
        'Sharpe Ratio':        round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        'Sortino Ratio':       round(sortino, 4) if not np.isnan(sortino) else np.nan,
        'Calmar Ratio':        round(calmar, 4) if not np.isnan(calmar) else np.nan,
        'Max Drawdown (%)':    round(max_dd * 100, 2),
        'Portfolio beta':      round(beta, 4) if not np.isnan(beta) else np.nan,
    }
