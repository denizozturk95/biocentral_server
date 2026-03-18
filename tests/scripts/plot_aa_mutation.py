#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Amino acids ordered by physicochemical property groups for nicer heatmaps
AA_ORDER = list("GAVLIMFWPSTCYNQHKRDE")
# Hydrophobic: G A V L I M F W
# Special:     P
# Polar:       S T C Y N Q
# Charged:     H K R D E

def _load_mutation_data(filename: str = "aa_mutation_sensitivity_uniref50.csv") -> pd.DataFrame | None:
    path = REPORTS_DIR / filename
    if not path.exists():
        print(f"File not found: {path}")
        return None
    return pd.read_csv(path)

def plot_heatmap():
    df = _load_mutation_data()
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, metric, label in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        # Pivot: rows = replacement_aa, cols = masking_ratio
        pivot = df.pivot_table(
            values=metric,
            index="replacement_aa",
            columns="masking_ratio",
            aggfunc="mean",
        )

        # Reorder rows
        ordered_aas = [aa for aa in AA_ORDER if aa in pivot.index]
        pivot = pivot.reindex(ordered_aas)

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=label)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c * 100)}%" for c in pivot.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10, fontfamily="monospace")

        ax.set_xlabel("Mutation Rate", fontsize=11)
        ax.set_ylabel("Replacement Amino Acid", fontsize=11)
        ax.set_title(f"Mutation Sensitivity — {label}", fontsize=12, fontweight="bold")

    plt.suptitle("AA Mutation Sensitivity — ESM2-T12-35M", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_plot("aa_mutation_heatmap")
    plt.close()

def plot_line_overlay():
    df_mut = _load_mutation_data()
    if df_mut is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        cmap = matplotlib.colormaps.get_cmap("tab20")

        for i, aa in enumerate(AA_ORDER):
            aa_data = df_mut[df_mut["replacement_aa"] == aa]
            if aa_data.empty:
                continue
            agg = aa_data.groupby("masking_ratio")[metric].mean().reset_index()
            ax.plot(
                agg["masking_ratio"] * 100, agg[metric],
                linewidth=1.2, alpha=0.7, color=cmap(i / 20),
                label=aa,
            )

        # X-masking reference (if available)
        x_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
        if x_path.exists():
            df_x = pd.read_csv(x_path)
            df_x["masking_ratio_parsed"] = (
                df_x["parameter"].str.extract(r"mask(\d+)%").astype(float) / 100
            )
            agg_x = df_x.groupby("masking_ratio_parsed")[metric].mean().reset_index()
            ax.plot(
                agg_x["masking_ratio_parsed"] * 100, agg_x[metric],
                linewidth=3, color="black", linestyle="--", label="X-masking",
                alpha=0.9,
            )

        ax.set_xlabel("Mutation Rate (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"AA Mutation Curves — {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    _save_plot("aa_mutation_line_overlay")
    plt.close()


def plot_line_overlay_per_aa():
    """One plot per amino acid: per-bin mutation curves + X-masking reference."""
    df_mut = _load_mutation_data()
    if df_mut is None:
        return

    # Load X-masking reference once
    x_agg = {}
    x_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
    if x_path.exists():
        df_x = pd.read_csv(x_path)
        df_x["masking_ratio_parsed"] = (
            df_x["parameter"].str.extract(r"mask(\d+)%").astype(float) / 100
        )
        for metric in ["cosine_distance", "l2_distance"]:
            x_agg[metric] = df_x.groupby("masking_ratio_parsed")[metric].mean().reset_index()

    has_bins = "bin" in df_mut.columns
    bins = sorted(df_mut["bin"].dropna().unique()) if has_bins else [None]
    colors_bin = ["#2E86AB", "#E94F37", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#795548"]

    for aa in AA_ORDER:
        aa_data = df_mut[df_mut["replacement_aa"] == aa]
        if aa_data.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, metric, ylabel in [
            (axes[0], "cosine_distance", "Cosine Distance"),
            (axes[1], "l2_distance", "L2 Distance"),
        ]:
            if has_bins:
                for i, bin_val in enumerate(bins):
                    bin_data = aa_data[aa_data["bin"] == bin_val]
                    if bin_data.empty:
                        continue
                    agg = bin_data.groupby("masking_ratio")[metric].mean().reset_index()
                    ax.plot(
                        agg["masking_ratio"] * 100, agg[metric],
                        marker="o", markersize=3, linewidth=1.5,
                        label=f"len={int(bin_val)}", color=colors_bin[i % len(colors_bin)],
                        alpha=0.8,
                    )
            else:
                agg = aa_data.groupby("masking_ratio")[metric].mean().reset_index()
                ax.plot(
                    agg["masking_ratio"] * 100, agg[metric],
                    marker="o", markersize=4, linewidth=2, color="#E94F37",
                    label=f"AA={aa}",
                )

            if metric in x_agg:
                agg_x = x_agg[metric]
                ax.plot(
                    agg_x["masking_ratio_parsed"] * 100, agg_x[metric],
                    linewidth=2.5, color="black", linestyle="--", label="X-masking",
                    alpha=0.9,
                )

            ax.set_xlabel("Mutation Rate (%)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"Mutation to {aa} — {ylabel}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.arange(0, 101, 10))

        plt.suptitle(f"AA Mutation Sensitivity — Replace with {aa}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save_plot(f"aa_mutation_line_{aa}")
        plt.close()


def plot_bar_ranking():
    df = _load_mutation_data()
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, label in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        # Filter to ~10% mutation rate
        subset = df[df["masking_ratio"].between(0.09, 0.11)]
        if subset.empty:
            subset = df[df["masking_ratio"].between(0.04, 0.16)]

        ranking = (
            subset.groupby("replacement_aa")[metric]
            .mean()
            .sort_values(ascending=False)
        )

        colors = []
        for aa in ranking.index:
            if aa in "GAVLIMFW":
                colors.append("#E94F37")  # Hydrophobic
            elif aa in "STCYNQ":
                colors.append("#4CAF50")  # Polar
            elif aa in "HKRDE":
                colors.append("#2E86AB")  # Charged
            else:
                colors.append("#9C27B0")  # Special (P)

        ax.bar(range(len(ranking)), ranking.values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(ranking)))
        ax.set_xticklabels(ranking.index, fontsize=11, fontfamily="monospace")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"Embedding Change at ~10% Mutation — {label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Legend for AA types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E94F37", label="Hydrophobic (GAVLIMFW)"),
            Patch(facecolor="#4CAF50", label="Polar (STCYNQ)"),
            Patch(facecolor="#2E86AB", label="Charged (HKRDE)"),
            Patch(facecolor="#9C27B0", label="Special (P)"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    _save_plot("aa_mutation_bar_ranking")
    plt.close()

def _plot_uniref50_per_bin_at_ratio(
    df: pd.DataFrame,
    target_ratio: float,
    available_aas: list,
    suffix: str,
):
    """Helper: per-bin mutation sensitivity at a single masking ratio."""
    bins = sorted(df["bin"].dropna().unique())
    colors_bin = ["#2E86AB", "#E94F37", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#795548"]

    # Tolerance for matching the target ratio
    tol = 0.005 if target_ratio <= 0.05 else 0.02
    subset = df[df["masking_ratio"].between(target_ratio - tol, target_ratio + tol)]
    if subset.empty:
        print(f"No data near masking_ratio={target_ratio}")
        return

    pct_label = f"{int(target_ratio * 100)}%"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        for i, bin_val in enumerate(bins):
            bin_data = subset[subset["bin"] == bin_val]
            if bin_data.empty:
                continue
            ranking = (
                bin_data.groupby("replacement_aa")[metric]
                .mean()
                .reindex(available_aas)
            )
            ax.plot(
                range(len(ranking)), ranking.values,
                marker="o", markersize=4, linewidth=1.5,
                label=f"bin={int(bin_val)}", color=colors_bin[i % len(colors_bin)],
                alpha=0.8,
            )

        ax.set_xticks(range(len(available_aas)))
        ax.set_xticklabels(available_aas, fontsize=10, fontfamily="monospace")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Replacement Amino Acid", fontsize=11)
        ax.set_title(f"Per-Bin Mutation Sensitivity at ~{pct_label} — {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(f"aa_mutation_per_bin_uniref50_{suffix}")
    plt.close()


def plot_uniref50_per_bin():
    """Per-bin (per-length) mutation sensitivity at 2%, 10%, and 70% mutation rates."""
    df = _load_mutation_data("aa_mutation_sensitivity_uniref50.csv")
    if df is None:
        return

    if "bin" not in df.columns:
        print("No 'bin' column in UniRef50 mutation data")
        return

    available_aas = [aa for aa in AA_ORDER if aa in df["replacement_aa"].unique()]

    for ratio, suffix in [(0.02, "2pct"), (0.10, "10pct"), (0.75, "70pct")]:
        _plot_uniref50_per_bin_at_ratio(df, ratio, available_aas, suffix)


def plot_uniref50_per_bin_mean_map():
    """Create readable per-bin mean-distance maps.

    For each metric, saves a 2-panel figure:
    - Left: absolute mean distances (AAs ranked by global sensitivity)
    - Right: row-wise z-score map to show per-AA bin patterns clearly
    """
    df = _load_mutation_data("aa_mutation_sensitivity_uniref50.csv")
    if df is None:
        return

    if "bin" not in df.columns:
        print("No 'bin' column in UniRef50 mutation data")
        return

    grouped = (
        df.groupby(["bin", "replacement_aa"])[["cosine_distance", "l2_distance"]]
        .mean()
        .reset_index()
    )
    grouped["mean_distance"] = grouped[["cosine_distance", "l2_distance"]].mean(axis=1)

    available_aas = [aa for aa in AA_ORDER if aa in grouped["replacement_aa"].unique()]
    bins = sorted(grouped["bin"].dropna().unique())

    metrics = [
        ("cosine_distance", "Cosine Distance", "YlOrRd"),
        ("l2_distance", "L2 Distance", "YlGnBu"),
        ("mean_distance", "Mean Distance", "magma"),
    ]

    for metric, label, cmap in metrics:
        pivot = grouped.pivot_table(
            values=metric,
            index="replacement_aa",
            columns="bin",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=available_aas, columns=bins)

        # Rank AAs by global mean so important rows are always near the top.
        aa_rank = pivot.mean(axis=1).sort_values(ascending=False).index
        pivot_ranked = pivot.reindex(aa_rank)

        row_mean = pivot_ranked.mean(axis=1)
        row_std = pivot_ranked.std(axis=1).replace(0, np.nan)
        pivot_z = pivot_ranked.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.25, 1]})

        flat_vals = pivot_ranked.values[np.isfinite(pivot_ranked.values)]
        if flat_vals.size > 0:
            vmin = np.quantile(flat_vals, 0.05)
            vmax = np.quantile(flat_vals, 0.95)
        else:
            vmin, vmax = None, None

        im_abs = axes[0].imshow(
            pivot_ranked.values,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im_abs, ax=axes[0], label=f"{label} (mean)")

        im_z = axes[1].imshow(
            pivot_z.values,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-2,
            vmax=2,
        )
        plt.colorbar(im_z, ax=axes[1], label="Row z-score")

        for i, ax in enumerate(axes):
            ax.set_yticks(range(len(pivot_ranked.index)))
            ax.set_yticklabels(pivot_ranked.index, fontsize=10, fontfamily="monospace")
            step = max(1, len(pivot_ranked.columns) // 12)
            tick_idx = list(range(0, len(pivot_ranked.columns), step))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(int(pivot_ranked.columns[j])) for j in tick_idx], fontsize=9)
            ax.set_xlabel("Sequence Length Bin", fontsize=11)
            ax.set_ylabel("Replacement Amino Acid", fontsize=11)
            ax.grid(False)

            # Value labels help readability when the matrix is not too dense.
            if i == 0 and pivot_ranked.shape[0] * pivot_ranked.shape[1] <= 200:
                for r in range(pivot_ranked.shape[0]):
                    for c in range(pivot_ranked.shape[1]):
                        val = pivot_ranked.iat[r, c]
                        if pd.notna(val):
                            ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=6, color="black")

        axes[0].set_title(f"Per-Bin Mutation Sensitivity ({label})\nAbsolute mean", fontsize=12, fontweight="bold")
        axes[1].set_title("Relative bin effect per AA\n(row-wise z-score)", fontsize=12, fontweight="bold")

        plt.suptitle(f"UniRef50 Per-Bin Mean Distance Map - {label}", fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        _save_plot(f"aa_mutation_per_bin_mean_map_{metric}")
        plt.close()


def plot_uniref50_per_bin_mean_profiles(top_n: int = 8):
    """Non-heatmap view of per-bin mean mutation sensitivity.

    For each metric:
    - Left panel: line profiles across bins for top-N most sensitive AAs.
    - Right panel: per-bin population summary across all AAs (mean, std, min-max).
    """
    df = _load_mutation_data("aa_mutation_sensitivity_uniref50.csv")
    if df is None:
        return

    if "bin" not in df.columns:
        print("No 'bin' column in UniRef50 mutation data")
        return

    grouped = (
        df.groupby(["bin", "replacement_aa"])[["cosine_distance", "l2_distance"]]
        .mean()
        .reset_index()
    )
    grouped["mean_distance"] = grouped[["cosine_distance", "l2_distance"]].mean(axis=1)

    metrics = [
        ("cosine_distance", "Cosine Distance", "#E94F37"),
        ("l2_distance", "L2 Distance", "#2E86AB"),
        ("mean_distance", "Mean Distance", "#5D3A9B"),
    ]

    for metric, label, accent in metrics:
        pivot = grouped.pivot_table(
            values=metric,
            index="replacement_aa",
            columns="bin",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=sorted(pivot.columns))

        aa_order = pivot.mean(axis=1).sort_values(ascending=False).index
        top_aas = aa_order[: min(top_n, len(aa_order))]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})

        cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, aa in enumerate(top_aas):
            y = pivot.loc[aa]
            axes[0].plot(
                y.index,
                y.values,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                alpha=0.9,
                color=cmap(i % 10),
                label=aa,
            )

        axes[0].set_title(
            f"Top {len(top_aas)} AA Profiles Across Length Bins - {label}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel("Sequence Length Bin", fontsize=11)
        axes[0].set_ylabel(f"{label} (mean)", fontsize=11)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(title="AA", ncol=2, fontsize=8, title_fontsize=9, loc="upper left")

        per_bin = grouped.groupby("bin")[metric].agg(["mean", "std", "min", "max"]).reset_index()
        x = per_bin["bin"].values
        mean_y = per_bin["mean"].values
        std_y = per_bin["std"].fillna(0.0).values
        min_y = per_bin["min"].values
        max_y = per_bin["max"].values

        axes[1].fill_between(x, min_y, max_y, color=accent, alpha=0.15, label="AA range (min-max)")
        axes[1].fill_between(x, mean_y - std_y, mean_y + std_y, color=accent, alpha=0.25, label="mean ± std")
        axes[1].plot(x, mean_y, color=accent, linewidth=2.4, marker="o", markersize=4, label="mean")

        axes[1].set_title("Per-Bin Summary Across All AAs", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Sequence Length Bin", fontsize=11)
        axes[1].set_ylabel(label, fontsize=11)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)

        plt.suptitle(f"UniRef50 Per-Bin Mutation Sensitivity (Non-Heatmap) - {label}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save_plot(f"aa_mutation_per_bin_mean_profiles_{metric}")
        plt.close()


def plot_mutation_vs_xmasking_comparison():
    """Compare AA mutation distances to X-masking reference across masking ratios.

    Shows mean distance across all 20 AAs vs X-masking, with individual AA range as shaded band.
    """
    df_mut = _load_mutation_data()
    if df_mut is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        # Aggregate all AAs: mean, min, max per masking ratio
        agg = df_mut.groupby("masking_ratio").agg(
            mean_val=(metric, "mean"),
            min_val=(metric, "min"),
            max_val=(metric, "max"),
            std_val=(metric, "std"),
        ).reset_index()

        x = agg["masking_ratio"] * 100

        ax.plot(x, agg["mean_val"], linewidth=2, color="#E94F37", label="AA Mutation (mean)", alpha=0.9)
        ax.fill_between(x, agg["min_val"], agg["max_val"], alpha=0.15, color="#E94F37", label="AA range (min–max)")

        # X-masking reference
        x_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
        if x_path.exists():
            df_x = pd.read_csv(x_path)
            if "masking_ratio" not in df_x.columns:
                df_x["masking_ratio"] = (
                    df_x["parameter"].str.extract(r"mask(\d+)%").astype(float) / 100
                )
            agg_x = df_x.groupby("masking_ratio")[metric].mean().reset_index()
            ax.plot(
                agg_x["masking_ratio"] * 100, agg_x[metric],
                linewidth=2.5, color="black", linestyle="--", label="X-masking",
                alpha=0.9,
            )

        ax.set_xlabel("Mutation Rate (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"AA Mutation vs X-Masking — {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    _save_plot("aa_mutation_vs_xmasking")
    plt.close()


def _save_plot(name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = REPORTS_DIR / f"{name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {png_path}")

if __name__ == "__main__":
    plot_heatmap()
    plot_line_overlay()
    plot_line_overlay_per_aa()
    plot_bar_ranking()
    plot_uniref50_per_bin()
    plot_uniref50_per_bin_mean_map()
    plot_uniref50_per_bin_mean_profiles()
    plot_mutation_vs_xmasking_comparison()
