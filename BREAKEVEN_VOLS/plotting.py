import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
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
    ax.axvline(median_be, color="black", linewidth=1.8, label=f"Median = {median_be:.2%}")
    ax.axvline(mean_be, color="tab:green", linestyle="--", linewidth=1.2, label=f"Mean = {mean_be:.2%}")

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
        median_be = np.median(sigma_be_array)
        mean_be = sigma_be_array.mean()
        ax.axvline(median_be, color="black", linewidth=1.8, label=f"Median = {median_be:.2%}")
        ax.axvline(mean_be, color="tab:red", linestyle="--", linewidth=1.2, label=f"Mean = {mean_be:.2%}")
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
    Box-and-whisker plot of breakeven vol by moneyness (the "smile") -- one
    box per option showing the full bootstrap distribution (median, IQR,
    whiskers, outlier points), with a red diamond marking the mean and a
    text label giving its exact value above each box.

    results_by_option -- dict of label -> {'sigma_be_array': array, 'moneyness': float}
    """
    items = sorted(results_by_option.items(), key=lambda kv: kv[1]['moneyness'])
    labels = [label for label, _ in items]
    moneyness = np.array([r['moneyness'] for _, r in items])
    data = [r['sigma_be_array'][~np.isnan(r['sigma_be_array'])] for _, r in items]
    medians = np.array([np.nanmedian(r['sigma_be_array']) for _, r in items])
    means = np.array([np.nanmean(r['sigma_be_array']) for _, r in items])
    tops = np.array([d.max() if len(d) else np.nan for d in data])

    spacing = np.min(np.diff(np.sort(moneyness))) if len(moneyness) > 1 else 0.05
    width = spacing * 0.6

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.boxplot(
        data, positions=moneyness, widths=width, patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="none", markeredgecolor="tab:red", markersize=6),
        medianprops=dict(color="black", linewidth=1.8),
        boxprops=dict(facecolor="tab:blue", alpha=0.5),
        whiskerprops=dict(color="tab:blue"),
        capprops=dict(color="tab:blue"),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    # median is the primary reported statistic (robust to the occasional
    # degenerate/near-zero draw at deep-OTM, short-dated cells); mean is
    # labeled separately beside its own diamond marker so it doesn't stack
    # the annotation tall enough to collide with the chart title
    for x, top, y_med, y_mean, label in zip(moneyness, tops, medians, means, labels):
        ax.annotate(f"{label}\nmedian={y_med:.2%}", (x, top), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color="black", fontweight="bold")
        ax.annotate(f"mean={y_mean:.2%}", (x, y_mean), textcoords="offset points", xytext=(24, 0),
                    ha="left", va="center", fontsize=7.5, color="tab:red")

    ax.set_xlim(moneyness.min() - width * 2, moneyness.max() + width * 2)
    ax.set_xticks(moneyness)
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


def plot_vol_surface_comparison(moneyness_levels, expiry_labels, modeled_surface, market_surface,
                                 highlight_points=None, title="Modeled Breakeven Vol vs Market Implied Vol",
                                 save_path=None):
    """
    Overlays the modeled breakeven-vol surface (green) and the market
    implied-vol surface (purple) on the same moneyness/expiry axes for
    direct visual comparison. Cells that are NaN in either surface (e.g. no
    listed contract matched, or a degenerate breakeven-vol cell) just leave
    a gap in that surface.
    """
    X, Y = np.meshgrid(moneyness_levels, np.arange(len(expiry_labels)))

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, np.asarray(modeled_surface), color="tab:green", alpha=0.55,
                     edgecolor="black", linewidth=0.2, antialiased=True)
    ax.plot_surface(X, Y, np.asarray(market_surface), color="tab:purple", alpha=0.55,
                     edgecolor="black", linewidth=0.2, antialiased=True)

    ax.set_xticks(moneyness_levels)
    ax.set_xticklabels([f"{m:.0%}" for m in moneyness_levels], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(expiry_labels)))
    ax.set_yticklabels(expiry_labels, fontsize=8)
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: f"{z:.0%}"))

    ax.set_xlabel("Moneyness (K/S)", labelpad=14)
    ax.set_ylabel("Expiry", labelpad=10)
    ax.set_zlabel("Volatility", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=22, azim=-60)

    legend_handles = [
        Patch(facecolor="tab:green", alpha=0.55, label="Modeled breakeven vol"),
        Patch(facecolor="tab:purple", alpha=0.55, label="Market implied vol"),
    ]
    if highlight_points:
        expiry_index = {label: idx for idx, label in enumerate(expiry_labels)}
        hx = [m for m, _, _ in highlight_points]
        hy = [expiry_index[label] for _, label, _ in highlight_points]
        hz = [v for _, _, v in highlight_points]
        ax.scatter(hx, hy, hz, color="red", s=70, depthshade=False, zorder=10)
        legend_handles.append(Patch(facecolor="red", label="Requested strikes"))

    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_variance_risk_premium_heatmap(moneyness_levels, expiry_labels, vrp_vol,
                                        title="Variance Risk Premium (Implied - Breakeven Vol)",
                                        save_path=None):
    """
    Heatmap of VRP in vol-point terms across the (expiry x moneyness) grid,
    diverging colormap centered at 0: positive (red) means market implied
    vol trades above modeled breakeven vol (the usual direction -- options
    priced with a premium over what historical hedging would have needed);
    negative (blue) means the reverse.
    """
    vrp_vol = np.asarray(vrp_vol)
    finite = vrp_vol[np.isfinite(vrp_vol)]
    vmax = np.max(np.abs(finite)) if finite.size else 0.01

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(vrp_vol, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(moneyness_levels)))
    ax.set_xticklabels([f"{m:.0%}" for m in moneyness_levels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(expiry_labels)))
    ax.set_yticklabels(expiry_labels)

    for i in range(vrp_vol.shape[0]):
        for j in range(vrp_vol.shape[1]):
            val = vrp_vol[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:+.1%}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > vmax * 0.5 else "black")

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Expiry")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Implied vol - Breakeven vol")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_smile_comparison_grid(moneyness_levels, expiry_labels, modeled_surface, market_surface,
                                title="Breakeven Vol vs Market Implied Vol", save_path=None):
    """
    Small-multiples grid, one panel per expiry, each a plain two-line smile
    comparison vs moneyness: modeled breakeven vol against market implied
    vol, on the same axes so the shapes are directly comparable.
    """
    n = len(expiry_labels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharex=True)
    axes = np.asarray(axes).flatten()

    for i, (ax, label) in enumerate(zip(axes, expiry_labels)):
        ax.plot(moneyness_levels, modeled_surface[i], color="tab:green", marker="o", markersize=4,
                linewidth=1.8, label="Breakeven vol (modeled, median)")
        ax.plot(moneyness_levels, market_surface[i], color="tab:purple", marker="s", markersize=4,
                linewidth=1.8, label="Implied vol (market)")
        ax.set_title(label)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Volatility")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, axes


def plot_variance_swap_gamma_weights(S_grid, gamma_naive_flat, gamma_correct_flat,
                                      gamma_naive_skew, gamma_correct_skew, F,
                                      title="Static Hedge Weighting: Naive (1/K) vs Correct (1/K²)",
                                      save_path=None):
    """
    Recreates Exhibits 2.2.2/2.2.3 from the JPMorgan variance swap note: the
    aggregate dollar gamma of the replicating strip under the 'naive' 1/K
    weighting (only regionally linear, not flat) versus the theoretically
    correct 1/K^2 weighting (constant across a wide region).

    Two panels, because the paper's clean flat-gamma result only holds under
    its own simplifying assumption of a single flat vol across every strike
    in the strip (used to isolate the pure weighting math in Exhibits
    2.2.1-2.2.3). Left panel reproduces that idealized case. Right panel
    uses each strike's own real listed (skewed) implied vol -- the actual
    hedge you'd trade -- showing that real skew erodes the idealized
    flatness of even the 'correct' 1/K^2 weighting. Each curve is normalized
    by its own value at the forward, since 1/K vs 1/K^2 differ in absolute
    scale by a factor of ~K and what matters here is shape, not level.
    """
    S_grid = np.asarray(S_grid)
    f_idx = int(np.argmin(np.abs(S_grid - F)))

    def norm(curve):
        curve = np.asarray(curve)
        return curve / curve[f_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, g_naive, g_correct, subtitle in [
        (ax1, gamma_naive_flat, gamma_correct_flat, "Idealized: single flat vol for every strike"),
        (ax2, gamma_naive_skew, gamma_correct_skew, "Real market: each strike's own listed (skewed) vol"),
    ]:
        ax.plot(S_grid, norm(g_naive), color="tab:red", linewidth=1.8, label="Naive weights (w ∝ 1/K)")
        ax.plot(S_grid, norm(g_correct), color="tab:green", linewidth=1.8, label="Correct weights (w ∝ 1/K²)")
        ax.axvline(F, color="gray", linestyle="--", linewidth=1, label=f"Forward = {F:.2f}")
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Underlying Level")
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Aggregate Dollar Gamma (normalized to 1 at the forward)")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, (ax1, ax2)


def plot_variance_swap_smile_and_strike(chain_df, F, K_var, K_var_rule_of_thumb, iv_90put,
                                         title="SPY Smile & Variance Swap Strike",
                                         save_path=None):
    """
    Scatter of the listed option chain's implied vol smile (puts vs calls,
    by moneyness), with the rigorously-computed replication fair strike, the
    Demeterfi-Derman-Kamal-Zou rule-of-thumb strike, and the 90% put IV
    (Section 1.1's informal shorthand for where the fair strike tends to
    land) all marked for comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # OTM side only per type (puts below the forward, calls at/above) --
    # matches the paper's own replication convention (Section 2.3) and
    # avoids double-plotting both wings once per type, which just overlays
    # two near-identical curves via put-call parity and is harder to read.
    puts = chain_df[(chain_df["option_type"] == "put") & (chain_df["strike"] < F)]
    calls = chain_df[(chain_df["option_type"] == "call") & (chain_df["strike"] >= F)]

    ax.scatter(puts["strike"] / F, puts["iv"], color="tab:red", s=14, alpha=0.6, label="Listed OTM puts")
    ax.scatter(calls["strike"] / F, calls["iv"], color="tab:blue", s=14, alpha=0.6, label="Listed OTM calls")

    ax.axhline(K_var, color="tab:green", linewidth=1.8, linestyle="-", label=f"Replication fair strike = {K_var:.1%}")
    ax.axhline(K_var_rule_of_thumb, color="tab:orange", linewidth=1.8, linestyle="--",
               label=f"Rule-of-thumb strike = {K_var_rule_of_thumb:.1%}")
    ax.scatter([0.90], [iv_90put], color="black", marker="D", s=50, zorder=5,
               label=f"90% put IV = {iv_90put:.1%}")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="Forward (K/F = 100%)")

    ax.set_xlabel("Moneyness (K / Forward)")
    ax.set_ylabel("Implied Volatility")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_variance_swap_hedge_pnl(days, price_path, F, cumulative_pnl, mtm_path, terminal_payoff,
                                  title="Static Replication + Daily Delta Hedge vs Theoretical MTM",
                                  save_path=None):
    """
    Two-panel figure: the bootstrapped price path with the forward marked
    (top), and the simulated static-strip + daily-delta-hedge cumulative
    P&L against the closed-form mark-to-market path from Section 1.3
    (bottom). If the replication is working, the simulated P&L should
    roughly track the theoretical MTM curve and land near the theoretical
    terminal payoff at expiry -- the gap between them is the real-world
    replication error (finite strikes, discrete hedging, discrete strip).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(days, price_path, color="tab:blue", linewidth=1.5, label="Simulated price path")
    ax1.axhline(F, color="gray", linestyle="--", linewidth=1, label=f"Forward = {F:.2f}")
    ax1.set_ylabel("Price")
    ax1.set_title("Underlying Path")
    ax1.legend(fontsize=8)

    ax2.plot(days, cumulative_pnl, color="tab:green", linewidth=1.6, label="Simulated hedge P&L")
    ax2.plot(days, mtm_path, color="tab:purple", linewidth=1.6, linestyle="--", label="Theoretical MTM (§1.3 formula)")
    ax2.axhline(terminal_payoff, color="black", linestyle=":", linewidth=1.2,
                label=f"Theoretical terminal payoff = {terminal_payoff:.4f}")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("P&L (variance points)")
    ax2.set_xlabel("Days")
    ax2.set_title("Hedge P&L vs Theoretical Mark-to-Market")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, (ax1, ax2)


def plot_variance_swap_sensitivities(days, vega_path, skew_sens_path,
                                      title="Variance Swap Sensitivities Over the Trade Life",
                                      save_path=None):
    """
    Section 1.3's Vega and skew sensitivity, both of which decay linearly
    to zero as (T-t)/T shrinks toward expiry -- a direct consequence of
    variance being additive in time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(days, vega_path, color="tab:blue", linewidth=1.6)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_ylabel("Vega (per unit variance notional)")
    ax1.set_title("Vega Sensitivity")

    ax2.plot(days, skew_sens_path, color="tab:red", linewidth=1.6)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Skew Sensitivity (∂MTM/∂skew)")
    ax2.set_xlabel("Days")
    ax2.set_title("Skew Sensitivity")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, (ax1, ax2)


def plot_variance_swap_wing_truncation(strip, truncation, F,
                                        title="Replication Coverage: Listed Strikes vs Theoretical Continuum",
                                        save_path=None):
    """
    Section 3.2/2.3 note that perfect replication needs infinitely many
    strikes from 0 to infinity, which a real listed chain can never supply.
    This plots each listed strike's contribution to the replication
    integral (bars) against the strike range actually available, annotated
    with the fraction of the theoretical (flat-vol-extrapolated) integral
    that range captures -- the uncaptured fraction is real, unhedgeable
    wing/gap risk.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    contribution = strip["mid"] * strip["weight"]
    colors = ["tab:red" if t == "put" else "tab:blue" for t in strip["option_type"]]
    ax.bar(strip["strike"], contribution, width=strip["strike"].diff().median() * 0.8, color=colors, alpha=0.7)
    ax.axvline(F, color="gray", linestyle="--", linewidth=1, label=f"Forward = {F:.2f}")

    k_min, k_max = truncation["strike_range"]
    ax.axvspan(k_min, k_max, color="tab:green", alpha=0.08, label="Listed strike range")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Contribution to Replication Integral (Price × Weight)")
    ax.set_title(f"{title}\nCaptured: {truncation['captured_fraction']:.1%} of theoretical continuum integral")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax


def plot_variance_swap_dividend_adjustment(K_var_ex_div, K_var_div_adj, div_info,
                                            title="Impact of Dividends on the Fair Strike",
                                            save_path=None):
    """
    Section 3.3: dividends bias realized variance upward, so the ex-dividend
    fair strike understates the true fair strike. Simple bar comparison of
    the two, annotated with the trailing dividend data used to compute the
    adjustment.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    bars = ax.bar(["Ex-dividend\n(naive)", "Dividend-adjusted"], [K_var_ex_div, K_var_div_adj],
                   color=["tab:gray", "tab:green"], width=0.5)
    for bar, val in zip(bars, [K_var_ex_div, K_var_div_adj]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2%}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Fair Strike (Volatility)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(
        f"{title}\nTrailing yield: {div_info['dividend_yield']:.2%} "
        f"({div_info['n_payments_trailing_year']} payments, ${div_info['annual_dividend']:.2f}/yr)"
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

    return fig, ax
