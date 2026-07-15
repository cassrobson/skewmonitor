
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from plotting import (
    plot_delta_hedge_pnl,
    plot_breakeven_convergence,
    plot_greeks_timeseries,
    plot_block_bootstrap_shading,
    plot_bootstrap_price_paths,
    plot_breakeven_vol_distribution_grid,
    plot_breakeven_vol_smile,
    plot_breakeven_vol_surface,
    plot_vol_surface_comparison,
    plot_variance_risk_premium_heatmap,
    plot_smile_comparison_grid,
)

ALPACA_API_KEY="PKSWP8A9QH87I5DX69Y4"
ALPACA_API_SECRET="we0Fo1a7UmijE45Z3rWYkZjXqnsfFg3B8pwrnbdC"
ALPACA_BASE_URL="https://paper-api.alpaca.markets/"

from alpaca.data.requests import OptionChainRequest
from alpaca.data.historical import OptionHistoricalDataClient

option_historical_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# Alpaca rate limit courtesy delay between chain requests (matches dispersion_alpaca.py)
_MAX_REQUESTS_PER_MINUTE = 150
_DELAY_PER_REQUEST = 60 / _MAX_REQUESTS_PER_MINUTE


def parse_option_symbol(symbol):
    """OCC option symbol -> (ticker, expiry_date, 'C'/'P', strike_price)."""
    match = re.match(r'([A-Z]+)(\d{6})([CP])(\d+)', symbol)
    if not match:
        return None, None, None, None
    ticker, expiry, option_type, strike = match.groups()
    expiry_date = pd.to_datetime('20' + expiry, format='%Y%m%d').date()
    strike_price = float(strike) / 1000
    return ticker, expiry_date, option_type, strike_price

class OptionContract:
    def __init__(self, S, K, T, r, option_type, dt=1 / 252):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration in years
        self.r = r  # Risk-free interest rate
        self.option_type = option_type  # 'call' or 'put'
        self.dt = dt  # Simulation step size in years

        # Working sigma is the full-sample realized vol -- a stable, regime-
        # agnostic scalar (see estimate_full_sample_realized_vol), used to
        # bracket the breakeven-vol search and as the fixed reference vol
        # in the bootstrap comparison plots
        self.sigma_hat = self.estimate_full_sample_realized_vol(dt=self.dt)
        self.sigma = self.sigma_hat
        self.option_price = self.calculate_price()

    def _d1_d2(self, S, T, sigma):
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def calculate_price(self, S=None, T=None, sigma=None):
        S = self.S if S is None else S
        T = self.T if T is None else T
        sigma = self.sigma if sigma is None else sigma
        d1, d2 = self._d1_d2(S, T, sigma)

        if self.option_type == 'call':
            return S * norm.cdf(d1) - self.K * np.exp(-self.r * T) * norm.cdf(d2)
        elif self.option_type == 'put':
            return self.K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def calculate_delta(self, S=None, T=None, sigma=None):
        S = self.S if S is None else S
        T = self.T if T is None else T
        sigma = self.sigma if sigma is None else sigma
        d1, _ = self._d1_d2(S, T, sigma)

        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def calculate_gamma(self, S=None, T=None, sigma=None):
        S = self.S if S is None else S
        T = self.T if T is None else T
        sigma = self.sigma if sigma is None else sigma
        d1, _ = self._d1_d2(S, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def calculate_theta(self, S=None, T=None, sigma=None):
        S = self.S if S is None else S
        T = self.T if T is None else T
        sigma = self.sigma if sigma is None else sigma
        d1, d2 = self._d1_d2(S, T, sigma)

        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if self.option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r * T) * norm.cdf(-d2)

        return term1 + term2  # per year

    def estimate_full_sample_realized_vol(self, dt=None):
        """
        Annualized realized vol computed from the *entire* historical
        log-return sample, rather than a single trailing-window snapshot.
        More stable/regime-agnostic than a rolling estimate taken at one
        point in time, and consistent with block-bootstrap sampling from
        the full historical distribution rather than just its recent tail.
        """
        dt = self.dt if dt is None else dt
        df = pd.read_csv('daily_close_prices.csv')
        log_returns = np.log(df['SPY'] / df['SPY'].shift(1)).dropna()
        return log_returns.std() * np.sqrt(1 / dt)

    def generate_gbm_price_path_with_initial_realized_vol_estimate(self, num_steps, dt):
        # Estimate annualized realized vol from the full historical sample
        sigma_hat = self.estimate_full_sample_realized_vol(dt=dt)

        # Simulate a GBM path forward from S using that vol estimate
        rng = np.random.default_rng()
        z = rng.standard_normal(num_steps)
        drift = (self.r - 0.5 * sigma_hat ** 2) * dt
        diffusion = sigma_hat * np.sqrt(dt) * z
        log_path = np.cumsum(drift + diffusion)

        price_path = np.empty(num_steps + 1)
        price_path[0] = self.S
        price_path[1:] = self.S * np.exp(log_path)

        return price_path, sigma_hat

    def generate_block_bootstrap_price_path(self, num_steps, block_size=21, log_returns=None, rng=None,
                                             candidate_starts=None):
        """
        Builds a synthetic forward price path by stitching together randomly
        drawn, contiguous blocks of *real* historical daily log returns
        (moving block bootstrap). Unlike IID resampling, this preserves each
        block's internal autocorrelation and volatility clustering.

        log_returns      -- optional pre-loaded array of historical log returns,
                             to avoid re-reading the CSV on every call
        rng              -- optional shared np.random.Generator for reproducibility
        candidate_starts -- optional array restricting which block-start indices
                             are eligible to be drawn (e.g. only indices that fall
                             in historically high-vol/"stress" periods -- see
                             compute_stress_block_starts). Defaults to every
                             possible start index when not given.

        Returns (price_path, block_starts) where block_starts are the
        starting indices (into log_returns) of each block drawn, in order.
        """
        if log_returns is None:
            df = pd.read_csv('daily_close_prices.csv')
            log_returns = np.log(df['SPY'] / df['SPY'].shift(1)).dropna().values
        else:
            log_returns = np.asarray(log_returns)

        if rng is None:
            rng = np.random.default_rng()

        n = len(log_returns)
        n_blocks = int(np.ceil(num_steps / block_size))
        if candidate_starts is not None and len(candidate_starts) > 0:
            block_starts = rng.choice(candidate_starts, size=n_blocks)
        else:
            block_starts = rng.integers(0, n - block_size, size=n_blocks)

        sampled_returns = np.concatenate(
            [log_returns[s:s + block_size] for s in block_starts]
        )[:num_steps]

        price_path = np.empty(num_steps + 1)
        price_path[0] = self.S
        price_path[1:] = self.S * np.exp(np.cumsum(sampled_returns))

        return price_path, block_starts

    def simulate_delta_hedge_pnl(self, price_path, dt, sigma_hedge):
        """
        Walks price_path day by day (vectorized), delta-hedging the option
        using Black-Scholes greeks computed at sigma_hedge, and returns the
        greek time series plus the cumulative hedge P&L.
        """
        S = np.asarray(price_path)
        n = len(S)
        T_arr = np.maximum(self.T - np.arange(n) * dt, 1e-6)  # floor to avoid division by zero at expiry

        option_values = self.calculate_price(S=S, T=T_arr, sigma=sigma_hedge)
        deltas = self.calculate_delta(S=S, T=T_arr, sigma=sigma_hedge)
        gammas = self.calculate_gamma(S=S, T=T_arr, sigma=sigma_hedge)
        thetas = self.calculate_theta(S=S, T=T_arr, sigma=sigma_hedge)

        # Long the option, short delta shares against it, rebalanced daily.
        # shares_held[i] is the hedge held during the interval [i, i+1).
        dS = np.diff(S)
        shares_held = -deltas[:-1]
        hedge_pnl = shares_held * dS
        option_pnl = np.diff(option_values)
        daily_pnl = option_pnl + hedge_pnl
        cumulative_pnl = np.concatenate(([0.0], np.cumsum(daily_pnl)))

        return {
            'deltas': deltas,
            'gammas': gammas,
            'thetas': thetas,
            'option_values': option_values,
            'cumulative_pnl': cumulative_pnl,
            'total_pnl': cumulative_pnl[-1],
        }

    def break_even_volatility(self, price_path, dt=None, sigma_bounds=None, max_expansions=6, degenerate_tol=1e-6):
        """
        Root-finds the hedging vol sigma such that total delta-hedge P&L
        over price_path is zero. price_path must be supplied explicitly
        (e.g. one bootstrap draw) since there's no longer a single canonical
        path on the option itself. Brackets the search around sigma_hat.

        Raises ValueError if no bracket can be found, OR if the option is
        degenerately worthless across the entire bracket (|P&L| < degenerate_tol
        at both endpoints) -- e.g. very short-dated, deep OTM combinations
        where the option value floors to exact 0.0 in float64 for any
        realistic sigma. In that regime brentq would otherwise "solve" for
        whatever sigma happens to cross a floating-point underflow boundary,
        which is numerical noise, not a meaningful breakeven vol.
        """
        dt = self.dt if dt is None else dt
        if sigma_bounds is None:
            sigma_bounds = (0.2 * self.sigma_hat, 3 * self.sigma_hat)

        def total_pnl(sigma):
            return self.simulate_delta_hedge_pnl(price_path, dt, sigma)['total_pnl']

        lo, hi = sigma_bounds
        f_lo, f_hi = total_pnl(lo), total_pnl(hi)

        expansions = 0
        while f_lo * f_hi > 0 and expansions < max_expansions:
            lo = max(lo / 2, 1e-4)
            hi = hi * 2
            f_lo, f_hi = total_pnl(lo), total_pnl(hi)
            expansions += 1

        if f_lo * f_hi > 0:
            raise ValueError(
                f"No sign change in hedge P&L found in sigma range [{lo:.4f}, {hi:.4f}] "
                f"after {expansions} expansions (P&L(lo)={f_lo:.4f}, P&L(hi)={f_hi:.4f})."
            )

        if max(abs(f_lo), abs(f_hi)) < degenerate_tol:
            raise ValueError(
                f"Option is degenerately worthless across the entire sigma range "
                f"[{lo:.4f}, {hi:.4f}] (|P&L| < {degenerate_tol}) -- no meaningful "
                f"breakeven vol exists (likely deep OTM / very short-dated)."
            )

        return brentq(total_pnl, lo, hi)


def compute_stress_block_starts(log_returns, block_size=21, vol_window=21, percentile=75, direction="down"):
    """
    Identifies which block-start indices fall in historically high-vol
    ("stress") periods, for restricting the block bootstrap to stress-only
    sampling. Defines "stress" as days where trailing `vol_window`-day
    annualized realized vol is at/above the given percentile of its own
    full-history distribution -- e.g. percentile=75 means "top quartile of
    realized vol regimes."

    High realized vol alone isn't the same as a selloff -- sharp relief
    rallies (e.g. the V-shaped bounce right after the 2020 COVID crash) are
    also high-vol but are the opposite of what "downside stress" means for
    a put-skew story. `direction` narrows the high-vol days further by the
    sign of the trailing `vol_window`-day return:
      "down" -- high vol AND a negative trailing return (selloffs/crashes only)
      "up"   -- high vol AND a positive trailing return (violent rallies only)
      "any"  -- high vol regardless of direction (the original behavior)

    A block's *start* day is used as the proxy for the whole block being in
    a stress regime -- since the trailing rolling window at the start index
    already overlaps most of the block itself, this is a reasonable, simple
    filter without needing to check every day in the block.

    Returns (candidate_starts, threshold, stress_frac) where candidate_starts
    is the array of eligible start indices, threshold is the vol cutoff used,
    and stress_frac is the fraction of all possible starts that qualified.
    """
    if direction not in ("down", "up", "any"):
        raise ValueError("direction must be 'down', 'up', or 'any'")

    returns = pd.Series(log_returns)
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    rolling_ret = returns.rolling(vol_window).sum()  # trailing cumulative log return
    threshold = np.nanpercentile(rolling_vol.dropna(), percentile)

    n = len(log_returns)
    max_start = n - block_size
    starts = np.arange(max_start)

    high_vol = rolling_vol.values[:max_start] >= threshold
    if direction == "down":
        directional = rolling_ret.values[:max_start] < 0
    elif direction == "up":
        directional = rolling_ret.values[:max_start] > 0
    else:
        directional = np.ones(max_start, dtype=bool)

    eligible = high_vol & directional
    candidate_starts = starts[eligible]

    return candidate_starts, threshold, len(candidate_starts) / max_start


def run_block_bootstrap_smile_analysis(options, dt, num_paths=200, block_size=21, seed=42,
                                        candidate_starts=None):
    """
    Draws num_paths block-bootstrapped price paths ONCE -- shared across every
    option in `options`, since they're all written on the same underlying --
    and solves the breakeven vol for each option against every path. Building
    the smile off one common set of realized paths (rather than redrawing
    independently per strike) keeps the comparison across strikes apples-to-
    apples. Keeps each option's first-path result as a "representative"
    sample for single-path diagnostic plots.

    candidate_starts -- optional array restricting block draws to specific
                         historical windows (e.g. stress-only periods, see
                         compute_stress_block_starts). Defaults to unrestricted.
    """
    df = pd.read_csv('daily_close_prices.csv')
    log_returns = np.log(df['SPY'] / df['SPY'].shift(1)).dropna().values

    base = options[0]
    num_steps = int(round(base.T / dt))
    rng = np.random.default_rng(seed)

    price_paths = np.empty((num_paths, num_steps + 1))
    block_starts_representative = None

    for i in range(num_paths):
        # generate_block_bootstrap_price_path only depends on S/history, not
        # on K/type, so it's safe to reuse across all options via `base`
        path, block_starts = base.generate_block_bootstrap_price_path(
            num_steps, block_size=block_size, log_returns=log_returns, rng=rng,
            candidate_starts=candidate_starts,
        )
        price_paths[i] = path
        if i == 0:
            block_starts_representative = block_starts

    results_by_option = {}
    for option in options:
        moneyness = option.K / base.S
        label = f"{moneyness:.0%} {option.option_type.capitalize()}"

        # NaN-guard: very short-dated/deep-OTM combinations can fail to
        # bracket a sign change (option value ~0 for any reasonable sigma),
        # which would otherwise raise and abort the whole sweep
        sigma_be_array = np.full(num_paths, np.nan)
        for i in range(num_paths):
            try:
                sigma_be_array[i] = option.break_even_volatility(price_path=price_paths[i], dt=dt)
            except ValueError:
                pass

        valid_idx = np.flatnonzero(~np.isnan(sigma_be_array))
        rep_idx = int(valid_idx[0]) if len(valid_idx) else 0
        representative_sigma_be = sigma_be_array[rep_idx]
        representative_hedge_result = (
            option.simulate_delta_hedge_pnl(price_paths[rep_idx], dt, representative_sigma_be)
            if len(valid_idx) else None
        )
        results_by_option[label] = {
            'option': option,
            'moneyness': moneyness,
            'sigma_be_array': sigma_be_array,
            'representative': {
                'price_path': price_paths[rep_idx],
                'sigma_be': representative_sigma_be,
                'hedge_result': representative_hedge_result,
            },
        }

    return {
        'price_paths': price_paths,
        'block_starts_representative': block_starts_representative,
        'block_size': block_size,
        'results_by_option': results_by_option,
    }


def run_block_bootstrap_surface_analysis(S0, r, dt, expiry_specs, moneyness_type_pairs,
                                          num_paths=300, block_size=21, seed=42,
                                          candidate_starts=None):
    """
    Sweeps a full (expiry x moneyness) grid. For each expiry, builds the full
    set of options at that expiry/rate and runs run_block_bootstrap_smile_analysis
    against one shared path ensemble for that expiry (comparisons across
    moneyness stay apples-to-apples within an expiry; each expiry draws its
    own path ensemble since path length differs). Cells where the root-find
    couldn't bracket a sign change come back as NaN in the per-option arrays
    (see run_block_bootstrap_smile_analysis) rather than aborting the sweep.

    candidate_starts -- optional array restricting block draws to specific
                         historical windows (e.g. stress-only periods, see
                         compute_stress_block_starts). Defaults to unrestricted.
    """
    expiry_labels = [label for label, _ in expiry_specs]
    moneyness_levels = np.array([m for m, _ in moneyness_type_pairs])
    n_expiries = len(expiry_specs)
    n_moneyness = len(moneyness_type_pairs)

    mean_surface = np.full((n_expiries, n_moneyness), np.nan)
    median_surface = np.full((n_expiries, n_moneyness), np.nan)
    std_surface = np.full((n_expiries, n_moneyness), np.nan)

    results_by_expiry = {}
    rng_master = np.random.default_rng(seed)

    for i, (expiry_label, T) in enumerate(expiry_specs):
        options = [
            OptionContract(S=S0, K=S0 * moneyness, T=T, r=r, option_type=opt_type, dt=dt)
            for moneyness, opt_type in moneyness_type_pairs
        ]
        expiry_seed = int(rng_master.integers(0, 2 ** 32 - 1))
        smile_result = run_block_bootstrap_smile_analysis(
            options, dt, num_paths=num_paths, block_size=block_size, seed=expiry_seed,
            candidate_starts=candidate_starts,
        )
        results_by_expiry[expiry_label] = smile_result

        n_failed = 0
        for j, option in enumerate(options):
            moneyness = option.K / S0
            label = f"{moneyness:.0%} {option.option_type.capitalize()}"
            arr = smile_result['results_by_option'][label]['sigma_be_array']
            mean_surface[i, j] = np.nanmean(arr)
            median_surface[i, j] = np.nanmedian(arr)
            std_surface[i, j] = np.nanstd(arr)
            n_failed += int(np.isnan(arr).sum())

        note = f" ({n_failed} failed root-finds out of {num_paths * n_moneyness})" if n_failed else ""
        print(f"  {expiry_label}: done{note}")

    return {
        'expiry_labels': expiry_labels,
        'moneyness_levels': moneyness_levels,
        'mean_surface': mean_surface,
        'median_surface': median_surface,
        'std_surface': std_surface,
        'results_by_expiry': results_by_expiry,
    }


def get_chain_snapshot(symbol, spot, expiry):
    chain_snap = option_historical_data_client.get_option_chain(
    OptionChainRequest(
        underlying_symbol=symbol,
        expiration_date=expiry,
        strike_price_gte=str(spot-5),
        strike_price_lte=str(spot+5),
    )
    )

    data_list = []
    for symbol, details in chain_snap.items():
        quote = details.latest_quote  # Access attributes directly
        trade = details.latest_trade
        greeks = details.greeks

        data_list.append({
            'symbol': symbol,
            'bid_price': quote.bid_price if quote else None,
            'ask_price': quote.ask_price if quote else None,
            'bid_size': quote.bid_size if quote else None,
            'ask_size': quote.ask_size if quote else None,
            'bid_exchange': quote.bid_exchange if quote else None,
            'ask_exchange': quote.ask_exchange if quote else None,
            'quote_timestamp': quote.timestamp if quote else None,
            'trade_price': trade.price if trade else None,
            'trade_size': trade.size if trade else None,
            'trade_exchange': trade.exchange if trade else None,
            'trade_timestamp': trade.timestamp if trade else None,
            'implied_volatility': details.implied_volatility,
            'delta': greeks.delta if greeks else None,
            'gamma': greeks.gamma if greeks else None,
            'theta': greeks.theta if greeks else None,
            'vega': greeks.vega if greeks else None,
            'rho': greeks.rho if greeks else None
        })

    # Convert list to DataFrame
    df = pd.DataFrame(data_list)
    df[['ticker', 'expiry_date', 'option_type', 'strike_price']] = df['symbol'].apply(parse_option_symbol).apply(pd.Series)
    return df


def fetch_market_implied_vol_surface(symbol, S0, expiry_specs, moneyness_type_pairs,
                                      today=None, strike_pad=0.05, expiry_window_days=15,
                                      long_dated_window_fraction=0.2):
    """
    Pulls a market-implied-vol surface from Alpaca's live option chain,
    matched onto the same (expiry, moneyness) grid as the modeled breakeven
    vol surface, so the two line up cell-for-cell for comparison.

    Real listed expirations are fixed calendar dates, not exact year
    fractions, so for each target expiry this searches a window around
    "today + T" for whichever listed expiration date actually falls closest
    to the target, then for each moneyness level picks the listed contract
    (put below 100%, call at/above) whose strike is closest to S0*moneyness.
    The window widens for longer-dated targets (max of expiry_window_days
    and long_dated_window_fraction * days-to-target) since SPY LEAPS are
    listed far more sparsely (often quarterly) than near-term monthlies --
    a fixed +/-15 days that works fine for a 1M target can easily miss the
    nearest listed expiration for a 2Y/3Y target. Cells with no matching
    data (no contracts found, or none with a quoted implied_volatility)
    come back as NaN rather than raising.
    """
    if today is None:
        today = datetime.now().date()

    moneyness_levels = np.array([m for m, _ in moneyness_type_pairs])
    expiry_labels = [label for label, _ in expiry_specs]
    n_expiries = len(expiry_specs)
    n_moneyness = len(moneyness_type_pairs)

    iv_surface = np.full((n_expiries, n_moneyness), np.nan)
    matched_strikes = np.full((n_expiries, n_moneyness), np.nan)
    matched_expirations = [None] * n_expiries

    strike_low = S0 * min(moneyness_levels) * (1 - strike_pad)
    strike_high = S0 * max(moneyness_levels) * (1 + strike_pad)

    for i, (expiry_label, T) in enumerate(expiry_specs):
        target_date = today + timedelta(days=round(T * 365))
        days_out = (target_date - today).days
        window = max(expiry_window_days, int(long_dated_window_fraction * days_out))

        try:
            chain = option_historical_data_client.get_option_chain(
                OptionChainRequest(
                    underlying_symbol=symbol,
                    expiration_date_gte=(target_date - timedelta(days=window)).isoformat(),
                    expiration_date_lte=(target_date + timedelta(days=window)).isoformat(),
                    strike_price_gte=str(round(strike_low, 2)),
                    strike_price_lte=str(round(strike_high, 2)),
                )
            )
        except Exception as exc:
            print(f"  {expiry_label}: chain fetch failed ({type(exc).__name__}: {exc})")
            time.sleep(_DELAY_PER_REQUEST)
            continue

        rows = []
        for occ_symbol, details in chain.items():
            _, exp_date, opt_type, strike = parse_option_symbol(occ_symbol)
            if exp_date is None or details.implied_volatility is None:
                continue
            rows.append({'expiry_date': exp_date, 'option_type': opt_type, 'strike': strike,
                         'iv': details.implied_volatility})

        if not rows:
            print(f"  {expiry_label}: no contracts with a quoted implied vol in range")
            time.sleep(_DELAY_PER_REQUEST)
            continue

        chain_df = pd.DataFrame(rows)
        chain_df['dte_diff'] = (pd.to_datetime(chain_df['expiry_date']) - pd.Timestamp(target_date)).abs()
        nearest_expiry = chain_df.loc[chain_df['dte_diff'].idxmin(), 'expiry_date']
        chain_df = chain_df[chain_df['expiry_date'] == nearest_expiry]
        matched_expirations[i] = nearest_expiry

        for j, (moneyness, opt_type) in enumerate(moneyness_type_pairs):
            target_strike = S0 * moneyness
            side = chain_df[chain_df['option_type'] == ('P' if opt_type == 'put' else 'C')]
            if side.empty:
                continue
            nearest_row = side.iloc[(side['strike'] - target_strike).abs().argmin()]
            iv_surface[i, j] = nearest_row['iv']
            matched_strikes[i, j] = nearest_row['strike']

        n_matched = int(np.sum(~np.isnan(iv_surface[i])))
        print(f"  {expiry_label}: matched to listed expiry {nearest_expiry} "
              f"({n_matched}/{n_moneyness} moneyness levels found)")
        time.sleep(_DELAY_PER_REQUEST)

    return {
        'expiry_labels': expiry_labels,
        'moneyness_levels': moneyness_levels,
        'iv_surface': iv_surface,
        'matched_strikes': matched_strikes,
        'matched_expirations': matched_expirations,
    }


def compute_variance_risk_premium(breakeven_surface, iv_surface):
    """
    VRP aligned cell-for-cell with both surfaces: vol-point terms
    (implied - breakeven, the more directly interpretable spread) and
    variance terms (implied^2 - breakeven^2, the textbook VRP definition).
    Positive means market implied vol trades above modeled breakeven vol --
    the usual direction, since implied vol embeds a risk premium for
    crash/tail risk that historically realized hedging outcomes don't
    fully justify.
    """
    vrp_vol = iv_surface - breakeven_surface
    vrp_variance = iv_surface ** 2 - breakeven_surface ** 2
    return vrp_vol, vrp_variance


def main():
    os.makedirs("plots", exist_ok=True)

    # --- Quick toggle: sample blocks only from historically high-vol
    # ("stress") periods instead of the full 10-year history. All outputs
    # get a "_stress" filename suffix so the normal-sampling plots are never
    # overwritten -- flip this, rerun, and compare the two sets side by side.
    STRESS_SAMPLING = True
    STRESS_PERCENTILE = 75   # "stress" = top quartile of trailing 21d realized vol
    STRESS_DIRECTION = "down"  # "down" = selloffs only, "up" = rallies only, "any" = either

    df = pd.read_csv("daily_close_prices.csv", index_col=0, parse_dates=True)
    S0 = df["SPY"].iloc[-1]
    r = 0.00
    dt = 1 / 252
    block_size = 21       # ~1 trading month, to preserve local vol clustering
    num_paths = 1000        # per (expiry, moneyness) cell -- 8 x 7 = 56 cells,
                            # so this already means ~56,000 root-finds; raise once
                            # you've confirmed a first pass looks right

    suffix = "_stress" if STRESS_SAMPLING else ""
    direction_label = {"down": "Down-Stress", "up": "Up-Stress", "any": "Stress"}[STRESS_DIRECTION]
    title_suffix = f" -- {direction_label} Sampling" if STRESS_SAMPLING else ""

    def out(name):
        return f"plots/{name}{suffix}.png"

    candidate_starts = None
    if STRESS_SAMPLING:
        log_returns_full = np.log(df["SPY"] / df["SPY"].shift(1)).dropna().values
        candidate_starts, stress_threshold, stress_frac = compute_stress_block_starts(
            log_returns_full, block_size=block_size, vol_window=21, percentile=STRESS_PERCENTILE,
            direction=STRESS_DIRECTION,
        )
        print(f"{direction_label}-only sampling: trailing 21d realized vol >= {stress_threshold:.1%} "
              f"({STRESS_PERCENTILE}th pct), direction={STRESS_DIRECTION} -- {len(candidate_starts)} "
              f"eligible block-starts ({stress_frac:.1%} of history)")

    expiry_specs = [
        ("1M", 30 / 365),
        ("2M", 60 / 365),
        ("3M", 91 / 365),
        ("6M", 182 / 365),
        ("1Y", 365 / 365),
        ("2Y", 730 / 365),
        ("3Y", 1095 / 365),
    ]

    # 80%..120% moneyness in 5% steps: puts below 100%, calls at/above 100%
    moneyness_levels = np.round(np.arange(80, 120, 5) / 100, 2)
    moneyness_type_pairs = [(m, 'put' if m < 1.0 else 'call') for m in moneyness_levels]

    print(f"Building breakeven vol surface ({direction_label.upper() if STRESS_SAMPLING else 'normal'} sampling): "
          f"{len(expiry_specs)} expiries x {len(moneyness_type_pairs)} moneyness levels, {num_paths} paths each...")
    surface_result = run_block_bootstrap_surface_analysis(
        S0, r, dt, expiry_specs, moneyness_type_pairs,
        num_paths=num_paths, block_size=block_size, seed=42,
        candidate_starts=candidate_starts,
    )

    # 1. The surface itself, with the 4 moneyness levels the manager
    #    originally asked for highlighted in red at the 1Y expiry
    requested_moneyness = [0.80, 0.90, 1.00, 1.10]
    moneyness_idx = {m: j for j, m in enumerate(surface_result['moneyness_levels'])}
    expiry_idx_1y = surface_result['expiry_labels'].index("1Y")
    highlight_points = [
        (m, "1Y", surface_result['median_surface'][expiry_idx_1y, moneyness_idx[m]])
        for m in requested_moneyness
    ]

    # Median is the primary reported statistic -- more robust than the mean
    # to the occasional degenerate/near-zero draw at deep-OTM, short-dated cells
    plot_breakeven_vol_surface(
        surface_result['moneyness_levels'], surface_result['expiry_labels'],
        surface_result['median_surface'], highlight_points=highlight_points,
        title=f"Breakeven Volatility Surface (Median){title_suffix}",
        save_path=out("08_breakeven_vol_surface"),
    )

    # Everything below reuses the 1Y row already computed as part of the
    # surface sweep -- no separate bootstrap run needed
    smile_result_1y = surface_result['results_by_expiry']["1Y"]
    T_1y = dict(expiry_specs)["1Y"]
    days = np.arange(int(round(T_1y / dt)) + 1) * dt * 365

    requested_labels = [f"{m:.0%} {'Put' if m < 1.0 else 'Call'}" for m in requested_moneyness]
    results_by_option_1y = {
        label: smile_result_1y['results_by_option'][label] for label in requested_labels
    }

    for label, res in results_by_option_1y.items():
        arr = res['sigma_be_array']
        print(f"{label} (1Y): mean={np.nanmean(arr):.4f}, median={np.nanmedian(arr):.4f}, std={np.nanstd(arr):.4f}")

    # 2. Block bootstrap mechanism (1Y expiry, shared across its 21 options)
    plot_block_bootstrap_shading(
        df.index, df["SPY"].values, smile_result_1y['block_starts_representative'], block_size,
        title=f"Block Bootstrap Sampling{title_suffix}",
        save_path=out("01_block_bootstrap_shading"),
    )

    # 3. Fan chart of the 1Y bootstrapped forward paths, requested strikes marked
    requested_strikes = [S0 * m for m in requested_moneyness]
    plot_bootstrap_price_paths(
        smile_result_1y['price_paths'], strikes=requested_strikes, dt=dt,
        title=f"Block-Bootstrapped Price Paths{title_suffix}",
        save_path=out("02_bootstrap_price_paths"),
    )

    # 4. Quad grid: breakeven vol distribution for the 4 requested options
    plot_breakeven_vol_distribution_grid(
        results_by_option_1y,
        title=f"Breakeven Volatility Distributions Across the Smile{title_suffix}",
        save_path=out("03_breakeven_vol_distribution_grid"),
    )

    # 5. The 1Y smile across just the 4 requested options, labeled
    plot_breakeven_vol_smile(
        results_by_option_1y,
        title=f"Breakeven Volatility Smile{title_suffix}",
        save_path=out("04_breakeven_vol_smile"),
    )

    # 6-8. Deep dive on the ATM call's representative sample: delta/PnL,
    #      greeks, and the root-finding curve that solved its breakeven vol
    atm_label = "100% Call"
    atm_option = results_by_option_1y[atm_label]['option']
    atm_rep = results_by_option_1y[atm_label]['representative']

    plot_delta_hedge_pnl(
        days, atm_rep['hedge_result']['deltas'], atm_rep['hedge_result']['cumulative_pnl'],
        title=f"{atm_label} (1Y): Delta Hedging Activity & Cumulative P&L{title_suffix}",
        save_path=out("05_atm_delta_hedge_pnl"),
    )

    plot_greeks_timeseries(
        days, atm_rep['hedge_result']['deltas'], atm_rep['hedge_result']['gammas'], atm_rep['hedge_result']['thetas'],
        title=f"{atm_label} (1Y): Greeks Over Time{title_suffix}",
        save_path=out("06_atm_greeks_timeseries"),
    )

    sigma_grid = np.linspace(0.3 * atm_option.sigma_hat, 2.5 * atm_option.sigma_hat, 50)
    pnl_grid = [
        atm_option.simulate_delta_hedge_pnl(atm_rep['price_path'], dt, s)['total_pnl'] for s in sigma_grid
    ]
    plot_breakeven_convergence(
        sigma_grid, pnl_grid, atm_rep['sigma_be'],
        title=f"{atm_label} (1Y): Breakeven Volatility Convergence{title_suffix}",
        save_path=out("07_atm_breakeven_convergence"),
    )

    # 9. Pull the market-implied vol surface on the same (expiry, moneyness)
    #    grid and compare against the modeled breakeven vol surface
    print("Fetching market implied vol surface from Alpaca...")
    market_result = fetch_market_implied_vol_surface(
        "SPY", S0, expiry_specs, moneyness_type_pairs,
    )

    plot_vol_surface_comparison(
        surface_result['moneyness_levels'], surface_result['expiry_labels'],
        surface_result['median_surface'], market_result['iv_surface'],
        highlight_points=highlight_points,
        title=f"Modeled Breakeven Vol (Median) vs Market Implied Vol{title_suffix}",
        save_path=out("09_vol_surface_comparison"),
    )

    # 10. Variance risk premium: implied vol minus modeled breakeven vol,
    #     cell by cell across the grid
    vrp_vol, vrp_variance = compute_variance_risk_premium(
        surface_result['median_surface'], market_result['iv_surface']
    )
    plot_variance_risk_premium_heatmap(
        surface_result['moneyness_levels'], surface_result['expiry_labels'], vrp_vol,
        title=f"Variance Risk Premium (Implied - Median Breakeven Vol){title_suffix}",
        save_path=out("10_variance_risk_premium"),
    )

    # 11. Flattened 2D version of the surface comparison: one panel per
    #     expiry, plain modeled-vs-market lines vs moneyness
    plot_smile_comparison_grid(
        surface_result['moneyness_levels'], surface_result['expiry_labels'],
        surface_result['median_surface'], market_result['iv_surface'],
        title=f"Breakeven Vol (Median) vs Market Implied Vol{title_suffix}",
        save_path=out("11_smile_comparison_grid"),
    )


if __name__ == "__main__":
    main()
