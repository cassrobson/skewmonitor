"""
Variance swap modeling, following JPMorgan's "Just What You Need to Know
About Variance Swaps" (Bossu, Strasser, Guichard, 2005) -- section numbers
in the docstrings below refer to that note.

Reuses the SPY dataset, block-bootstrap path machinery, and Alpaca wiring
already built in run_analysis.py rather than duplicating them.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from run_analysis import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    OptionContract,
    option_historical_data_client,
    parse_option_symbol,
    _DELAY_PER_REQUEST,
)
from alpaca.data.requests import OptionChainRequest, CorporateActionsRequest
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.enums import CorporateActionsType

from plotting import (
    plot_variance_swap_gamma_weights,
    plot_variance_swap_smile_and_strike,
    plot_variance_swap_hedge_pnl,
    plot_variance_swap_sensitivities,
    plot_variance_swap_wing_truncation,
    plot_variance_swap_dividend_adjustment,
)

corporate_actions_client = CorporateActionsClient(ALPACA_API_KEY, ALPACA_API_SECRET)


# ---------------------------------------------------------------------------
# Black-Scholes pricing with a dividend yield q (vectorized over strike/time)
# -- run_analysis.py's OptionContract has no q term, so this stays separate
# rather than risking a regression on the already-reviewed breakeven-vol
# pipeline. Needed here because the replication strip prices dozens of
# strikes at once, and because forward != spot once dividends are included
# (Section 3.3).
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, q, sigma):
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, q, sigma, option_type):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return np.where(np.asarray(option_type) == "call", call, put)


def bs_delta(S, K, T, r, q, sigma, option_type):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    call_delta = disc_q * norm.cdf(d1)
    put_delta = disc_q * (norm.cdf(d1) - 1)
    return np.where(np.asarray(option_type) == "call", call_delta, put_delta)


def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, T, r, q, sigma, option_type):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    term1 = -(S * disc_q * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    call_term2 = -r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    put_term2 = r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    return term1 + np.where(np.asarray(option_type) == "call", call_term2, put_term2)


def forward_price(S0, r, q, T):
    return S0 * np.exp((r - q) * T)


# ---------------------------------------------------------------------------
# Dividends (Section 3.3)
# ---------------------------------------------------------------------------

def fetch_trailing_dividend_yield(symbol, S0, lookback_days=400):
    """
    Trailing 12-month cash dividend yield from Alpaca's corporate actions
    endpoint, feeding the continuous/discrete dividend-yield model in
    Section 3.3. Fetches a slightly wider window than 365 days so a payment
    right at the boundary isn't dropped, then filters back down to exactly
    the trailing year before summing.
    """
    end = date.today()
    start = end - timedelta(days=lookback_days)
    result = corporate_actions_client.get_corporate_actions(
        CorporateActionsRequest(symbols=[symbol], types=[CorporateActionsType.CASH_DIVIDEND],
                                 start=start, end=end)
    )
    divs = result.data.get("cash_dividends", [])
    cutoff = end - timedelta(days=365)
    trailing = [d.rate for d in divs if d.ex_date >= cutoff]

    total = float(sum(trailing))
    n_payments = len(trailing) if trailing else 4  # SPY pays quarterly; fall back if the window missed all of them

    return {
        "annual_dividend": total,
        "dividend_yield": total / S0,
        "n_payments_trailing_year": len(trailing),
        "div_per_payment": total / len(trailing) if trailing else 0.0,
        "n_payments_for_adjustment": n_payments,
    }


def dividend_adjusted_strike(k_var_ex_div, div_yield, n_divs_per_year):
    """Section 3.3, p.10: K_var ≈ sqrt((K_var^ex-div)^2 + DivYield^2 / NbDivsPerYear)."""
    return np.sqrt(k_var_ex_div ** 2 + (div_yield ** 2) / n_divs_per_year)


def variance_swap_dividend_sensitivity(notional, div_yield, n_divs_per_year, t, T, K_var):
    """Section 3.3, p.10: mu = dVarSwap/dDivYield rule of thumb."""
    return notional * (div_yield / n_divs_per_year) / K_var * (T - t) / T


# ---------------------------------------------------------------------------
# Option chain fetch -- full width, single nearest listed expiry. Unlike
# run_analysis.py's fetch_market_implied_vol_surface (a sparse 8-point
# comparison grid), replication needs as many actually-listed OTM strikes
# as possible.
# ---------------------------------------------------------------------------

def fetch_full_option_chain(symbol, S0, target_days_out, window_days=15,
                             strike_multiple_low=0.5, strike_multiple_high=1.5):
    today = date.today()
    target_date = today + timedelta(days=target_days_out)

    chain = option_historical_data_client.get_option_chain(
        OptionChainRequest(
            underlying_symbol=symbol,
            expiration_date_gte=(target_date - timedelta(days=window_days)).isoformat(),
            expiration_date_lte=(target_date + timedelta(days=window_days)).isoformat(),
            strike_price_gte=str(round(S0 * strike_multiple_low, 2)),
            strike_price_lte=str(round(S0 * strike_multiple_high, 2)),
        )
    )

    rows = []
    for occ_symbol, details in chain.items():
        _, exp_date, opt_type, strike = parse_option_symbol(occ_symbol)
        if exp_date is None:
            continue
        quote = details.latest_quote
        rows.append({
            "expiry_date": exp_date,
            "option_type": "put" if opt_type == "P" else "call",
            "strike": strike,
            "iv": details.implied_volatility,
            "bid": quote.bid_price if quote else None,
            "ask": quote.ask_price if quote else None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No {symbol} option chain data returned near {target_date}")

    df["dte_diff"] = (pd.to_datetime(df["expiry_date"]) - pd.Timestamp(target_date)).abs()
    nearest_expiry = df.loc[df["dte_diff"].idxmin(), "expiry_date"]
    df = df[df["expiry_date"] == nearest_expiry].drop(columns="dte_diff")

    df["mid"] = (df["bid"] + df["ask"]) / 2

    # Liquidity/quality filter -- real two-sided market and a plausible IV.
    # Deep ITM contracts especially throw erratic "implied vols" off wide,
    # stale quotes; dropping them here is also fine because replication only
    # ever uses the OTM side of each strike anyway (see select_otm_strip).
    n_before = len(df)
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"]) & df["iv"].notna() & (df["iv"] > 0.01) & (df["iv"] < 2.0)]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  dropped {n_dropped}/{n_before} contracts on liquidity/IV sanity filters")

    T = (pd.Timestamp(nearest_expiry).date() - today).days / 365
    return df.sort_values("strike").reset_index(drop=True), nearest_expiry, T


def _nearest_iv(chain_df, target_strike, option_type):
    side = chain_df[chain_df["option_type"] == option_type]
    if side.empty:
        return np.nan
    row = side.iloc[(side["strike"] - target_strike).abs().argmin()]
    return row["iv"]


def estimate_linear_skew(chain_df, F, low_moneyness=0.90, high_moneyness=1.00):
    """
    Slope of the skew curve w.r.t. forward moneyness (dSigma/d(K/F)), used by
    the Section 1.1 rule-of-thumb strike formula. Matches the paper's own
    "90-100 skew of 2 vegas" convention: IV(90%) - IV(100%) over a moneyness
    step of 0.10.
    """
    iv_low = _nearest_iv(chain_df, F * low_moneyness, "put" if low_moneyness < 1 else "call")
    iv_high = _nearest_iv(chain_df, F * high_moneyness, "call")
    return (iv_low - iv_high) / (high_moneyness - low_moneyness)


def rule_of_thumb_strike(atm_forward_vol, T, skew):
    """Demeterfi-Derman-Kamal-Zou (1999), Section 1.1 p.4."""
    return atm_forward_vol * np.sqrt(1 + 3 * T * skew ** 2)


# ---------------------------------------------------------------------------
# Static replication (Section 2.2/2.3): weight each OTM option by
# deltaK / K^power. power=2 is the theoretically-correct constant-dollar-
# gamma weight; power=1 reproduces the paper's "naive" weighting so the two
# can be compared directly (Exhibits 2.2.1-2.2.3).
# ---------------------------------------------------------------------------

def select_otm_strip(chain_df, F):
    """OTM puts (K<F) and OTM calls (K>=F) -- Section 2.3's k0=0-at-the-forward convention."""
    puts = chain_df[(chain_df["option_type"] == "put") & (chain_df["strike"] < F)].sort_values("strike")
    calls = chain_df[(chain_df["option_type"] == "call") & (chain_df["strike"] >= F)].sort_values("strike")
    return puts, calls


def _strike_deltas(strikes):
    """Local discretization step per strike (Exhibit 2.3.1's deltaK, generalized
    to non-uniform real listed strikes rather than an even 5% grid)."""
    strikes = np.asarray(strikes, dtype=float)
    if len(strikes) < 2:
        return np.zeros_like(strikes)
    return np.gradient(strikes)


def compute_replication_weights(chain_df, F, weight_power=2):
    puts, calls = select_otm_strip(chain_df, F)
    strip = pd.concat([puts, calls]).sort_values("strike").reset_index(drop=True)
    dK = _strike_deltas(strip["strike"].values)
    strip = strip.assign(dK=dK, weight=dK / (strip["strike"].values ** weight_power))
    return strip


def compute_fair_variance_strike(chain_df, F, T, r):
    """
    Section 2.3, p.17: VarSwap0 ~= (2/T) * sum[price(K)/K^2 * dK] - PV(T)*K_var^2,
    for a variance notional of 1. Solving VarSwap0 = 0 for K_var gives the
    fair strike -- the rigorous replacement for the 90%-put rule of thumb.
    """
    PV_T = np.exp(-r * T)
    strip = compute_replication_weights(chain_df, F, weight_power=2)
    raw_integral = float((strip["mid"] * strip["weight"]).sum())
    k_var_sq = (2.0 / T) * raw_integral / PV_T
    return {"fair_strike_variance": np.sqrt(k_var_sq), "raw_integral": raw_integral, "strip": strip}


def aggregate_dollar_gamma_curve(strip, S_grid, T, r, q, flat_vol=None):
    """
    Dollar gamma = Gamma * S^2 (footnote 9, p.14): second-order sensitivity
    to a *percent* change in the underlying, not an absolute one.

    flat_vol -- if given, every strike is greeked at this single vol instead
    of its own listed IV. This isolates the pure effect of the weighting
    scheme (w ∝ 1/K vs 1/K^2) on aggregate gamma shape, matching how the
    paper's Exhibits 2.2.1-2.2.3 are actually constructed (a single vol
    across the whole strip). Omit it to use each strike's own market-skew
    vol, which is what you'd really be hedging with.
    """
    strikes = strip["strike"].values[:, None]
    ivs = np.full_like(strikes, flat_vol, dtype=float) if flat_vol is not None else strip["iv"].values[:, None]
    weights = strip["weight"].values[:, None]
    S_row = np.asarray(S_grid)[None, :]
    gammas = bs_gamma(S_row, strikes, T, r, q, ivs)
    dollar_gamma = gammas * S_row ** 2
    return (weights * dollar_gamma).sum(axis=0)


# ---------------------------------------------------------------------------
# Wing truncation (Section 3.2/2.3): perfect replication needs strikes from
# 0 to infinity; a real listed chain has a hard lower/upper bound. Quantify
# how much of the theoretical integral that finite range actually captures.
# ---------------------------------------------------------------------------

def compute_wing_truncation(strip, F, T, r, atm_vol, extension_multiple=20, n_points=5000):
    """
    Uses a *flat* extrapolated vol (atm_vol) beyond the listed strike range
    as a simple reference scale for the "true" unbounded integral -- this is
    not a market read of the far wings (no skew extrapolation model is
    assumed), just a way to size how much of the replication integral's
    mass plausibly sits outside what's actually tradable.
    """
    k_min, k_max = strip["strike"].min(), strip["strike"].max()

    y_puts = np.linspace(F / extension_multiple, F, n_points)
    y_calls = np.linspace(F, F * extension_multiple, n_points)

    put_prices = bs_price(F, y_puts, T, r, 0.0, atm_vol, "put")
    call_prices = bs_price(F, y_calls, T, r, 0.0, atm_vol, "call")

    put_integrand = put_prices / y_puts ** 2
    call_integrand = call_prices / y_calls ** 2

    total_integral = np.trapezoid(put_integrand, y_puts) + np.trapezoid(call_integrand, y_calls)

    in_range_puts = y_puts >= k_min
    in_range_calls = y_calls <= k_max
    captured_integral = (
        np.trapezoid(put_integrand[in_range_puts], y_puts[in_range_puts])
        + np.trapezoid(call_integrand[in_range_calls], y_calls[in_range_calls])
    )

    return {
        "strike_range": (k_min, k_max),
        "captured_fraction": captured_integral / total_integral,
        "total_integral": total_integral,
        "captured_integral": captured_integral,
    }


# ---------------------------------------------------------------------------
# Mark-to-market & sensitivities (Section 1.3)
# ---------------------------------------------------------------------------

def variance_swap_mtm(notional, PV_t_T, t, T, realized_vol_0_t, implied_vol_t_T, strike):
    """p.8: MTM decomposition between realized (0,t) and implied (t,T) variance."""
    return notional * PV_t_T * ((t / T) * realized_vol_0_t ** 2 + ((T - t) / T) * implied_vol_t_T ** 2 - strike ** 2)


def variance_swap_vega(notional, implied_vol, t, T):
    """p.9: linear decay to zero as a direct consequence of variance's time-additivity."""
    return notional * (2 * implied_vol) * (T - t) / T


def variance_swap_skew_sensitivity(notional, atm_forward_vol, t, T):
    """p.9 rule of thumb (valid only for near-linear skew, per the paper's own caveat)."""
    return 6 * notional * atm_forward_vol ** 2 * (T - t) / T


# ---------------------------------------------------------------------------
# Hedge simulation: buy the static strip ONCE, then delta-hedge the
# AGGREGATE position daily with the underlying. The options position is
# static (Section 2.2); the P&L only converges to realized-vs-implied
# variance because of the ongoing daily delta-hedge (Section 2.1, Eq. 4/5)
# -- holding the basket unhedged would just be a basket of options, not a
# variance swap replication.
# ---------------------------------------------------------------------------

def simulate_variance_swap_static_hedge(strip, price_path, dt, r, q, T0, variance_notional=1.0):
    S = np.asarray(price_path)
    n = len(S)
    T_arr = np.maximum(T0 - np.arange(n) * dt, 1e-6)

    strikes = strip["strike"].values[:, None]
    ivs = strip["iv"].values[:, None]
    types = strip["option_type"].values[:, None]
    positions = (variance_notional * 2.0 / T0) * strip["weight"].values[:, None]

    S_row = S[None, :]
    T_row = T_arr[None, :]

    prices = bs_price(S_row, strikes, T_row, r, q, ivs, types)
    deltas = bs_delta(S_row, strikes, T_row, r, q, ivs, types)
    gammas = bs_gamma(S_row, strikes, T_row, r, q, ivs)
    thetas = bs_theta(S_row, strikes, T_row, r, q, ivs, types)

    portfolio_value = (positions * prices).sum(axis=0)
    portfolio_delta = (positions * deltas).sum(axis=0)
    portfolio_gamma = (positions * gammas).sum(axis=0)
    portfolio_theta = (positions * thetas).sum(axis=0)

    dS = np.diff(S)
    shares_held = -portfolio_delta[:-1]
    hedge_pnl = shares_held * dS
    option_pnl = np.diff(portfolio_value)
    daily_pnl = option_pnl + hedge_pnl
    cumulative_pnl = np.concatenate(([0.0], np.cumsum(daily_pnl)))

    return {
        "portfolio_value": portfolio_value,
        "portfolio_delta": portfolio_delta,
        "portfolio_gamma": portfolio_gamma,
        "portfolio_theta": portfolio_theta,
        "cumulative_pnl": cumulative_pnl,
        "total_pnl": cumulative_pnl[-1],
    }


def main():
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("daily_close_prices.csv", index_col=0, parse_dates=True)
    S0 = df["SPY"].iloc[-1]
    r = 0.00
    dt = 1 / 252
    T_target_days = 91  # ~3M, matching the paper's own worked-example tenor scale

    print(f"S0 = {S0:.2f}")

    print("Fetching SPY trailing dividend yield...")
    div_info = fetch_trailing_dividend_yield("SPY", S0)
    q = div_info["dividend_yield"]
    print(f"  trailing yield: {q:.2%} ({div_info['n_payments_trailing_year']} payments, "
          f"${div_info['annual_dividend']:.2f}/share/yr)")

    print("Fetching full SPY option chain...")
    chain_df, expiry_date, T = fetch_full_option_chain("SPY", S0, T_target_days)
    print(f"  matched to listed expiry {expiry_date} (T={T:.4f}y), {len(chain_df)} contracts after filtering")

    F = forward_price(S0, r, q, T)
    print(f"  forward = {F:.2f} (spot = {S0:.2f}, no-dividend forward would just be spot = {S0:.2f})")

    # --- Fair strike: rigorous replication vs 90%-put rule of thumb ---
    fair = compute_fair_variance_strike(chain_df, F, T, r)
    K_var = fair["fair_strike_variance"]
    strip_correct = fair["strip"]

    atm_vol = _nearest_iv(chain_df, F, "call")
    skew = estimate_linear_skew(chain_df, F)
    K_var_rot = rule_of_thumb_strike(atm_vol, T, skew)
    iv_90put = _nearest_iv(chain_df, F * 0.90, "put")

    print(f"  ATM-forward vol:            {atm_vol:.2%}")
    print(f"  90-100 skew slope:          {skew:.4f}")
    print(f"  Fair strike (replication):  {K_var:.2%}")
    print(f"  Rule-of-thumb strike:       {K_var_rot:.2%}")
    print(f"  90% put IV:                 {iv_90put:.2%}")

    K_var_div_adj = dividend_adjusted_strike(K_var, q, div_info["n_payments_for_adjustment"])
    print(f"  Dividend-adjusted strike:   {K_var_div_adj:.2%}")

    plot_variance_swap_smile_and_strike(
        chain_df, F, K_var, K_var_rot, iv_90put,
        save_path="plots/12_variance_swap_smile_and_strike.png",
    )
    plot_variance_swap_dividend_adjustment(
        K_var, K_var_div_adj, div_info,
        save_path="plots/13_variance_swap_dividend_adjustment.png",
    )

    # --- Naive (1/K) vs correct (1/K^2) weighting: recreate Exhibits 2.2.1-2.2.3 ---
    strip_naive = compute_replication_weights(chain_df, F, weight_power=1)
    S_grid = np.linspace(F * 0.5, F * 1.5, 400)
    gamma_naive_flat = aggregate_dollar_gamma_curve(strip_naive, S_grid, T, r, q, flat_vol=atm_vol)
    gamma_correct_flat = aggregate_dollar_gamma_curve(strip_correct, S_grid, T, r, q, flat_vol=atm_vol)
    gamma_naive_skew = aggregate_dollar_gamma_curve(strip_naive, S_grid, T, r, q)
    gamma_correct_skew = aggregate_dollar_gamma_curve(strip_correct, S_grid, T, r, q)
    plot_variance_swap_gamma_weights(
        S_grid, gamma_naive_flat, gamma_correct_flat, gamma_naive_skew, gamma_correct_skew, F,
        save_path="plots/14_variance_swap_gamma_weights.png",
    )

    # --- Wing truncation ---
    truncation = compute_wing_truncation(strip_correct, F, T, r, atm_vol)
    print(f"  Captured fraction of theoretical variance integral: {truncation['captured_fraction']:.1%} "
          f"(listed strikes {truncation['strike_range'][0]:.0f}-{truncation['strike_range'][1]:.0f})")
    plot_variance_swap_wing_truncation(
        strip_correct, truncation, F,
        save_path="plots/15_variance_swap_wing_truncation.png",
    )

    # --- Representative path: static replication + daily delta hedge ---
    base = OptionContract(S=S0, K=S0, T=T, r=r, option_type="call", dt=dt)  # reused only for path generation
    num_steps = int(round(T / dt))
    log_returns = np.log(df["SPY"] / df["SPY"].shift(1)).dropna().values
    rng = np.random.default_rng(7)
    price_path, _ = base.generate_block_bootstrap_price_path(num_steps, block_size=21, log_returns=log_returns, rng=rng)

    hedge = simulate_variance_swap_static_hedge(strip_correct, price_path, dt, r, q, T)

    days = np.arange(num_steps + 1) * dt * 365
    t_arr = days / 365

    daily_returns = np.diff(np.log(price_path))
    cum_sq = np.concatenate(([0.0], np.cumsum(daily_returns ** 2)))
    n_obs = np.arange(0, num_steps + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        realized_vol_cum = np.sqrt(252 * cum_sq / np.maximum(n_obs, 1))
    realized_vol_cum[0] = 0.0

    PV_t_T = np.exp(-r * (T - t_arr))
    implied_vol_path = np.full(num_steps + 1, atm_vol)  # flat term structure assumption -- simplification, no vol-of-vol model
    mtm_path = variance_swap_mtm(1.0, PV_t_T, t_arr, T, realized_vol_cum, implied_vol_path, K_var)

    realized_var_T = realized_vol_cum[-1] ** 2
    terminal_payoff = 1.0 * (realized_var_T - K_var ** 2)

    print(f"  Realized vol over simulated path: {realized_vol_cum[-1]:.2%}")
    print(f"  Simulated hedge terminal P&L:      {hedge['total_pnl']:.4f}")
    print(f"  Theoretical terminal payoff:       {terminal_payoff:.4f}")

    plot_variance_swap_hedge_pnl(
        days, price_path, F, hedge["cumulative_pnl"], mtm_path, terminal_payoff,
        save_path="plots/16_variance_swap_hedge_pnl.png",
    )

    vega_path = variance_swap_vega(1.0, atm_vol, t_arr, T)
    skew_sens_path = variance_swap_skew_sensitivity(1.0, atm_vol, t_arr, T)
    plot_variance_swap_sensitivities(
        days, vega_path, skew_sens_path,
        save_path="plots/17_variance_swap_sensitivities.png",
    )


if __name__ == "__main__":
    main()
