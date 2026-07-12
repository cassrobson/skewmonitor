import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection


def plot_price_path(price_path, strikes=None, dt=1 / 252, title="Simulated GBM Price Path", save_path=None):
    """
    Plots a simulated price path, optionally marking strike levels as
    horizontal reference lines.

    price_path -- array-like of prices, index 0 is spot at t=0
    strikes    -- optional list of strike prices to mark on the plot
    dt         -- time step size in years, used to build the day axis
    save_path  -- optional file path to save the figure to (e.g. "plots/path.png")
    """
    price_path = np.asarray(price_path)
    days = np.arange(len(price_path)) * dt * 365

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, price_path, color="tab:blue", linewidth=1.5, label="Price path")

    if strikes:
        colors = plt.cm.tab10.colors
        for i, k in enumerate(strikes):
            ax.axhline(k, color=colors[i % len(colors)], linestyle="--", linewidth=1, label=f"K = {k:.2f}")

    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_delta_hedge_pnl(days, deltas, cumulative_pnl, title="Delta Hedging Activity & Cumulative P&L", save_path=None):
    """
    Two-panel figure: the delta hedge ratio over time (top) and the
    cumulative delta-hedge P&L over time (bottom).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(days, deltas, color="tab:orange", linewidth=1.5)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_ylabel("Delta")
    ax1.set_title("Delta Hedge Ratio Over Time")

    ax2.plot(days, cumulative_pnl, color="tab:green", linewidth=1.5)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.set_xlabel("Days")
    ax2.set_title("Cumulative Delta-Hedge P&L")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, (ax1, ax2)


def plot_breakeven_convergence(sigma_grid, pnl_grid, sigma_be, title="Breakeven Volatility Convergence", save_path=None):
    """
    Plots total delta-hedge P&L over the option's life as a function of the
    hedging volatility sigma, with the solved breakeven vol marked where the
    curve crosses zero.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sigma_grid, pnl_grid, color="tab:blue", linewidth=1.5, label="Total hedge P&L(sigma)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(sigma_be, color="tab:red", linewidth=1.2, linestyle="--", label=f"Breakeven vol = {sigma_be:.2%}")
    ax.set_xlabel("Hedging volatility (sigma)")
    ax.set_ylabel("Total P&L over option life ($)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_greeks_timeseries(days, deltas, gammas, thetas, title="Greeks Over Time", save_path=None):
    """
    Three-panel figure showing delta, gamma, and theta over the life of the option.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(days, deltas, color="tab:orange")
    ax1.set_ylabel("Delta")

    ax2.plot(days, gammas, color="tab:purple")
    ax2.set_ylabel("Gamma")

    ax3.plot(days, thetas, color="tab:red")
    ax3.set_ylabel("Theta (per year)")
    ax3.set_xlabel("Days")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, (ax1, ax2, ax3)


def plot_block_bootstrap_shading(dates, prices, block_starts, block_size, title="Block Bootstrap Sampling", save_path=None):
    """
    Plots the full historical price series and shades the historical date
    windows (blocks) that were stitched together, in draw order, to build
    one block-bootstrapped path.
    """
    dates = np.asarray(dates)
    prices = np.asarray(prices)
    n = len(dates)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, color="tab:blue", linewidth=1, alpha=0.8, label="Historical SPY")

    colors = plt.cm.tab20.colors
    for i, start in enumerate(block_starts):
        end = min(start + block_size, n - 1)
        ax.axvspan(dates[start], dates[end], color=colors[i % len(colors)], alpha=0.35)
        ax.annotate(
            str(i + 1), xy=(dates[start], 0.96), xycoords=("data", "axes fraction"),
            fontsize=7, ha="left", va="top",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{title} (numbers show stitching order of one sampled path, block size = {block_size}d)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_bootstrap_price_paths(price_paths, strikes=None, dt=1 / 252, max_paths_shown=100,
                                title="Block-Bootstrapped Price Paths", save_path=None):
    """
    Spaghetti/fan chart of many block-bootstrapped forward price paths, with
    a median path and optional strike lines overlaid.
    """
    price_paths = np.asarray(price_paths)
    days = np.arange(price_paths.shape[1]) * dt * 365

    fig, ax = plt.subplots(figsize=(10, 6))

    n_show = min(max_paths_shown, price_paths.shape[0])
    idx = np.random.default_rng(0).choice(price_paths.shape[0], size=n_show, replace=False)

    for i in idx:
        ax.plot(days, price_paths[i], color="tab:blue", alpha=0.08, linewidth=0.8)

    median_path = np.median(price_paths, axis=0)
    ax.plot(days, median_path, color="tab:blue", linewidth=2, label="Median path")

    if strikes:
        colors = plt.cm.tab10.colors
        for i, k in enumerate(strikes):
            ax.axhline(k, color=colors[i % len(colors)], linestyle="--", linewidth=1.2, label=f"K = {k:.2f}")

    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_title(f"{title} ({n_show} of {price_paths.shape[0]} shown)")
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_breakeven_vol_distribution(sigma_be_array, sigma_hat=None,
                                     title="Breakeven Volatility Distribution (Block Bootstrap)", save_path=None):
    """
    Histogram of breakeven vols solved across bootstrap samples, with the
    realized-vol estimate marked for reference.
    """
    sigma_be_array = np.asarray(sigma_be_array)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sigma_be_array, bins=30, color="tab:blue", alpha=0.75, edgecolor="white")

    mean_be = np.mean(sigma_be_array)
    median_be = np.median(sigma_be_array)
    ax.axvline(mean_be, color="tab:green", linewidth=1.5, label=f"Mean = {mean_be:.2%}")
    ax.axvline(median_be, color="tab:orange", linewidth=1.5, label=f"Median = {median_be:.2%}")

    if sigma_hat is not None:
        ax.axvline(sigma_hat, color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"Realized vol estimate = {sigma_hat:.2%}")

    ax.set_xlabel("Breakeven volatility")
    ax.set_ylabel(f"Count (of {len(sigma_be_array)} bootstrap samples)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_hedge_pnl_fan(days, cumulative_pnl_matrix, max_paths_shown=100,
                        title="Delta-Hedge P&L Across Bootstrap Samples (fixed hedging vol)", save_path=None):
    """
    Fan chart of cumulative delta-hedge P&L trajectories across bootstrap
    samples, all hedged at the same fixed sigma -- shows how much P&L
    dispersion results from real path variation alone.
    """
    cumulative_pnl_matrix = np.asarray(cumulative_pnl_matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_show = min(max_paths_shown, cumulative_pnl_matrix.shape[0])
    idx = np.random.default_rng(1).choice(cumulative_pnl_matrix.shape[0], size=n_show, replace=False)

    for i in idx:
        ax.plot(days, cumulative_pnl_matrix[i], color="tab:green", alpha=0.08, linewidth=0.8)

    median_pnl = np.median(cumulative_pnl_matrix, axis=0)
    ax.plot(days, median_pnl, color="tab:green", linewidth=2, label="Median P&L")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"{title} ({n_show} of {cumulative_pnl_matrix.shape[0]} shown)")
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_breakeven_vol_distribution_grid(results_by_option,
                                          title="Breakeven Volatility Distributions Across the Smile", save_path=None):
    """
    2x2 grid of breakeven-vol histograms, one panel per option, on a shared
    x-axis scale so the spread/shift across strikes is directly comparable.

    results_by_option -- dict of label -> {'sigma_be_array': array, ...},
                          e.g. as returned by run_block_bootstrap_smile_analysis
    """
    labels = list(results_by_option.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = np.asarray(axes).flatten()

    all_values = np.concatenate([r['sigma_be_array'] for r in results_by_option.values()])
    all_values = all_values[~np.isnan(all_values)]
    bins = np.linspace(all_values.min(), all_values.max(), 30)

    for ax, label in zip(axes, labels):
        sigma_be_array = results_by_option[label]['sigma_be_array']
        sigma_be_array = sigma_be_array[~np.isnan(sigma_be_array)]
        ax.hist(sigma_be_array, bins=bins, color="tab:blue", alpha=0.75, edgecolor="white")
        mean_be = sigma_be_array.mean()
        ax.axvline(mean_be, color="tab:red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_be:.2%}")
        ax.set_title(label)
        ax.set_xlabel("Breakeven volatility")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for ax in axes[len(labels):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, axes


def plot_breakeven_vol_smile(results_by_option, title="Breakeven Volatility Smile", save_path=None):
    """
    Mean breakeven vol vs moneyness across the option set (the "smile"),
    with error bars showing +/- 1 std across bootstrap samples and each
    point labeled with its strike/type and vol level.

    results_by_option -- dict of label -> {'sigma_be_array': array, 'moneyness': float}
    """
    items = sorted(results_by_option.items(), key=lambda kv: kv[1]['moneyness'])
    labels = [label for label, _ in items]
    moneyness = np.array([r['moneyness'] for _, r in items])
    means = np.array([np.nanmean(r['sigma_be_array']) for _, r in items])
    stds = np.array([np.nanstd(r['sigma_be_array']) for _, r in items])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(moneyness, means, yerr=stds, fmt="o-", color="tab:blue", ecolor="tab:blue",
                elinewidth=1, capsize=4, markersize=7)

    for x, y, label in zip(moneyness, means, labels):
        ax.annotate(f"{label}\n{y:.2%}", (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)

    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Breakeven volatility")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_breakeven_vol_surface(moneyness_levels, expiry_labels, surface, highlight_points=None,
                                title="Breakeven Volatility Surface", save_path=None):
    """
    3D surface of breakeven vol across moneyness (x) and expiry (y). Expiry
    uses an ordinal axis with custom tick labels rather than raw time-to-
    expiry, since expiries here span 1 week to 3 years and would otherwise
    crush the short end together on a linear scale.

    moneyness_levels -- 1D array of moneyness values (the x grid)
    expiry_labels     -- list of expiry labels in the same order as `surface`'s rows
    surface           -- 2D array, shape (len(expiry_labels), len(moneyness_levels))
    highlight_points  -- optional list of (moneyness, expiry_label, vol) to mark in red,
                          e.g. specific strikes/expiry someone asked for
    """
    X, Y = np.meshgrid(moneyness_levels, np.arange(len(expiry_labels)))
    Z = np.asarray(surface)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, color="tab:green", alpha=0.6, edgecolor="black", linewidth=0.3, antialiased=True)

    ax.set_xticks(moneyness_levels)
    ax.set_xticklabels([f"{m:.0%}" for m in moneyness_levels], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(expiry_labels)))
    ax.set_yticklabels(expiry_labels, fontsize=8)
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: f"{z:.0%}"))

    ax.set_xlabel("Moneyness (K/S)", labelpad=14)
    ax.set_ylabel("Expiry", labelpad=10)
    ax.set_zlabel("Breakeven volatility", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=22, azim=-60)

    if highlight_points:
        expiry_index = {label: idx for idx, label in enumerate(expiry_labels)}
        hx = [m for m, _, _ in highlight_points]
        hy = [expiry_index[label] for _, label, _ in highlight_points]
        hz = [v for _, _, v in highlight_points]
        ax.scatter(hx, hy, hz, color="red", s=70, depthshade=False, label="Requested strikes", zorder=10)
        ax.legend(loc="upper left")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax
