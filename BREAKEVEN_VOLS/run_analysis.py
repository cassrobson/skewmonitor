
import os

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
)


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

    def generate_block_bootstrap_price_path(self, num_steps, block_size=21, log_returns=None, rng=None):
        """
        Builds a synthetic forward price path by stitching together randomly
        drawn, contiguous blocks of *real* historical daily log returns
        (moving block bootstrap). Unlike IID resampling, this preserves each
        block's internal autocorrelation and volatility clustering.

        log_returns -- optional pre-loaded array of historical log returns,
                       to avoid re-reading the CSV on every call
        rng         -- optional shared np.random.Generator for reproducibility

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


def run_block_bootstrap_smile_analysis(options, dt, num_paths=200, block_size=21, seed=42):
    """
    Draws num_paths block-bootstrapped price paths ONCE -- shared across every
    option in `options`, since they're all written on the same underlying --
    and solves the breakeven vol for each option against every path. Building
    the smile off one common set of realized paths (rather than redrawing
    independently per strike) keeps the comparison across strikes apples-to-
    apples. Keeps each option's first-path result as a "representative"
    sample for single-path diagnostic plots.
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
            num_steps, block_size=block_size, log_returns=log_returns, rng=rng
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
                                          num_paths=300, block_size=21, seed=42):
    """
    Sweeps a full (expiry x moneyness) grid. For each expiry, builds the full
    set of options at that expiry/rate and runs run_block_bootstrap_smile_analysis
    against one shared path ensemble for that expiry (comparisons across
    moneyness stay apples-to-apples within an expiry; each expiry draws its
    own path ensemble since path length differs). Cells where the root-find
    couldn't bracket a sign change come back as NaN in the per-option arrays
    (see run_block_bootstrap_smile_analysis) rather than aborting the sweep.
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
            options, dt, num_paths=num_paths, block_size=block_size, seed=expiry_seed
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


def main():
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("daily_close_prices.csv", index_col=0, parse_dates=True)
    S0 = df["SPY"].iloc[-1]
    r = 0.00
    dt = 1 / 252
    block_size = 21       # ~1 trading month, to preserve local vol clustering
    num_paths = 1000        # per (expiry, moneyness) cell -- 21 x 10 = 210 cells,
                            # so this already means ~63,000 root-finds; raise once
                            # you've confirmed a first pass looks right

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

    print(f"Building breakeven vol surface: {len(expiry_specs)} expiries x "
          f"{len(moneyness_type_pairs)} moneyness levels, {num_paths} paths each...")
    surface_result = run_block_bootstrap_surface_analysis(
        S0, r, dt, expiry_specs, moneyness_type_pairs,
        num_paths=num_paths, block_size=block_size, seed=42,
    )

    # 1. The surface itself, with the 4 moneyness levels the manager
    #    originally asked for highlighted in red at the 1Y expiry
    requested_moneyness = [0.80, 0.90, 1.00, 1.10]
    moneyness_idx = {m: j for j, m in enumerate(surface_result['moneyness_levels'])}
    expiry_idx_1y = surface_result['expiry_labels'].index("1Y")
    highlight_points = [
        (m, "1Y", surface_result['mean_surface'][expiry_idx_1y, moneyness_idx[m]])
        for m in requested_moneyness
    ]

    plot_breakeven_vol_surface(
        surface_result['moneyness_levels'], surface_result['expiry_labels'],
        surface_result['mean_surface'], highlight_points=highlight_points,
        save_path="plots/08_breakeven_vol_surface.png",
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
        save_path="plots/01_block_bootstrap_shading.png",
    )

    # 3. Fan chart of the 1Y bootstrapped forward paths, requested strikes marked
    requested_strikes = [S0 * m for m in requested_moneyness]
    plot_bootstrap_price_paths(
        smile_result_1y['price_paths'], strikes=requested_strikes, dt=dt,
        save_path="plots/02_bootstrap_price_paths.png",
    )

    # 4. Quad grid: breakeven vol distribution for the 4 requested options
    plot_breakeven_vol_distribution_grid(
        results_by_option_1y,
        save_path="plots/03_breakeven_vol_distribution_grid.png",
    )

    # 5. The 1Y smile across just the 4 requested options, labeled
    plot_breakeven_vol_smile(
        results_by_option_1y,
        save_path="plots/04_breakeven_vol_smile.png",
    )

    # 6-8. Deep dive on the ATM call's representative sample: delta/PnL,
    #      greeks, and the root-finding curve that solved its breakeven vol
    atm_label = "100% Call"
    atm_option = results_by_option_1y[atm_label]['option']
    atm_rep = results_by_option_1y[atm_label]['representative']

    plot_delta_hedge_pnl(
        days, atm_rep['hedge_result']['deltas'], atm_rep['hedge_result']['cumulative_pnl'],
        title=f"{atm_label} (1Y): Delta Hedging Activity & Cumulative P&L",
        save_path="plots/05_atm_delta_hedge_pnl.png",
    )

    plot_greeks_timeseries(
        days, atm_rep['hedge_result']['deltas'], atm_rep['hedge_result']['gammas'], atm_rep['hedge_result']['thetas'],
        title=f"{atm_label} (1Y): Greeks Over Time",
        save_path="plots/06_atm_greeks_timeseries.png",
    )

    sigma_grid = np.linspace(0.3 * atm_option.sigma_hat, 2.5 * atm_option.sigma_hat, 50)
    pnl_grid = [
        atm_option.simulate_delta_hedge_pnl(atm_rep['price_path'], dt, s)['total_pnl'] for s in sigma_grid
    ]
    plot_breakeven_convergence(
        sigma_grid, pnl_grid, atm_rep['sigma_be'],
        title=f"{atm_label} (1Y): Breakeven Volatility Convergence",
        save_path="plots/07_atm_breakeven_convergence.png",
    )


if __name__ == "__main__":
    main()
