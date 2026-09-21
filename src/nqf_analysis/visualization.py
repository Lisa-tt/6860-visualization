"""Create complementary static figures for the analytical narrative."""

from pathlib import Path
import textwrap

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import box

from .config import (
    OFFERING_COLORS,
    OFFERING_ORDER,
    RATING_COLORS,
    RATING_ORDER,
    REMOTENESS_ORDER,
    STATE_ORDER,
    output_paths,
)
from .spatial import build_equal_area_service_grid


REMOTENESS_SHORT = {
    "Major Cities of Australia": "Major Cities",
    "Inner Regional Australia": "Inner Regional",
    "Outer Regional Australia": "Outer Regional",
    "Remote Australia": "Remote",
    "Very Remote Australia": "Very Remote",
}


def _configure_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#374151",
            "text.color": "#111827",
            "axes.edgecolor": "#D1D5DB",
            "grid.color": "#E5E7EB",
            "savefig.facecolor": "white",
        }
    )


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _clean_axis(ax, grid_axis="x"):
    ax.grid(axis=grid_axis)
    ax.grid(axis="y" if grid_axis == "x" else "x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)


def _stacked_composition(ax, table, group_column, order, title):
    pivot = table.pivot(
        index=group_column, columns="rating", values="share_of_rated_pct"
    ).reindex(index=order, columns=RATING_ORDER, fill_value=0)
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for rating in RATING_ORDER:
        values = pivot[rating].fillna(0).to_numpy()
        ax.barh(y, values, left=left, color=RATING_COLORS[rating], label=rating)
        left += values
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of rated services (%)")
    ax.set_title(title, loc="left")
    _clean_axis(ax, "x")
    ax.spines["left"].set_visible(False)


def _kilometre_tick(value, _position):
    if value >= 1000:
        return f"{value / 1000:g}k km"
    return f"{value:g} km"


def plot_spatial_landscape(data, boundaries, path):
    grid = build_equal_area_service_grid(data)
    states = boundaries.dissolve(by="STE_NAME21").to_crs("EPSG:3577")
    extent = gpd.GeoSeries([box(110, -45, 155, -9)], crs="EPSG:4326").to_crs(
        "EPSG:3577"
    ).total_bounds

    fig, ax = plt.subplots(figsize=(12.8, 8.2))
    states.plot(
        ax=ax,
        facecolor="#F8FAFC",
        edgecolor="#475569",
        linewidth=0.7,
        zorder=1,
    )
    scatter = ax.scatter(
        grid["projected_x"],
        grid["projected_y"],
        s=np.maximum(8, grid["services"] * 1.35),
        c=grid["median_transport_km"].clip(lower=0.1, upper=1000),
        cmap="YlOrRd",
        norm=LogNorm(vmin=0.1, vmax=1000),
        alpha=0.76,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.015)
    colorbar.set_label("Median distance to nearest listed transport")
    colorbar.set_ticks([0.1, 1, 10, 100, 1000])
    colorbar.set_ticklabels(["0.1 km", "1 km", "10 km", "100 km", "1,000 km"])

    for count in [25, 100, 400]:
        ax.scatter(
            [],
            [],
            s=count * 1.35,
            facecolor="#F4A261",
            edgecolor="white",
            label=f"{count} services",
        )
    ax.legend(
        title="Services per 75 km grid cell",
        loc="lower left",
        frameon=False,
        labelspacing=1.2,
    )
    ax.set_title(
        "Approved services are concentrated along coastal urban corridors",
        loc="left",
        fontsize=18,
        pad=14,
    )
    fig.text(
        0.105,
        0.025,
        "Observed provision only: bubble area represents registered services, not places per child or unmet demand. ABS 2021 state and remoteness boundaries.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_service_offering_portfolio(tables, path):
    summary = tables["offering_summary"].sort_values("services")
    combinations = tables["offering_combinations"].head(9).sort_values("services")
    wrapped = [
        textwrap.fill(str(label), width=43)
        for label in combinations["offering_combination"]
    ]

    fig, (ax_offer, ax_combo) = plt.subplots(1, 2, figsize=(16, 7.3), gridspec_kw={"wspace": 0.48})
    colors = [OFFERING_COLORS[label] for label in summary["offering"]]
    bars = ax_offer.barh(summary["offering"], summary["services"], color=colors)
    for bar, row in zip(bars, summary.itertuples()):
        ax_offer.text(
            bar.get_width() + summary["services"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{row.services:,} ({row.share_all_services_pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax_offer.set_xlim(0, summary["services"].max() * 1.24)
    ax_offer.set_xlabel("Services recording the offering")
    ax_offer.set_title("Detailed service offerings", loc="left")
    _clean_axis(ax_offer, "x")
    ax_offer.spines["left"].set_visible(False)

    ax_combo.barh(wrapped, combinations["services"], color="#64748B")
    for y, count in enumerate(combinations["services"]):
        ax_combo.text(count + combinations["services"].max() * 0.015, y, f"{int(count):,}", va="center", fontsize=9)
    ax_combo.set_xlim(0, combinations["services"].max() * 1.16)
    ax_combo.set_xlabel("Services")
    ax_combo.set_title("Most common offering combinations", loc="left")
    _clean_axis(ax_combo, "x")
    ax_combo.spines["left"].set_visible(False)

    fig.suptitle("Appendix: detailed offerings and common combinations", x=0.06, y=1.01, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.06,
        0.01,
        "Detailed offerings are multi-label. One service can appear in several bars, so offering counts must not be added as market shares.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_offering_mix_by_remoteness(tables, path):
    table = tables["offering_by_remoteness"]
    gaps = table.pivot(index="offering", columns="remoteness_area", values="gap_from_national_pp").reindex(
        index=OFFERING_ORDER, columns=REMOTENESS_ORDER
    )
    counts = table.pivot(index="offering", columns="remoteness_area", values="services").reindex(
        index=OFFERING_ORDER, columns=REMOTENESS_ORDER
    )
    annotations = np.empty(gaps.shape, dtype=object)
    for row in range(gaps.shape[0]):
        for col in range(gaps.shape[1]):
            annotations[row, col] = f"{gaps.iloc[row, col]:+.1f}\n(n={int(counts.iloc[row, col]):,})"
    max_abs = max(2, float(np.nanmax(np.abs(gaps.to_numpy()))))
    fig, ax = plt.subplots(figsize=(13.8, 6.8))
    sns.heatmap(
        gaps,
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Gap from national offering prevalence (pp)"},
        ax=ax,
    )
    ax.set_xticklabels([REMOTENESS_SHORT[item] for item in REMOTENESS_ORDER], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Service mix changes with remoteness and may confound raw comparisons", loc="left", pad=16, fontsize=18)
    fig.text(
        0.125,
        0.01,
        "Cells show percentage-point differences from each offering's national prevalence; counts are included to expose small remote samples.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_quality(tables, path):
    fig = plt.figure(figsize=(16, 10.2))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.05, 1.1], hspace=0.38, wspace=0.28
    )
    ax_state = fig.add_subplot(grid[0, 0])
    ax_remote = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[:, 1])

    _stacked_composition(
        ax_state,
        tables["state_rating_composition"],
        "state",
        STATE_ORDER,
        "Overall rating composition by jurisdiction",
    )
    _stacked_composition(
        ax_remote,
        tables["remoteness_rating_composition"],
        "remoteness_area",
        REMOTENESS_ORDER,
        "Overall rating composition by remoteness",
    )
    ax_remote.set_yticklabels([REMOTENESS_SHORT[item] for item in REMOTENESS_ORDER])
    handles, labels = ax_state.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))

    benchmark_table = tables["quality_benchmark_by_state"]
    benchmark = benchmark_table.pivot(index="state", columns="measure", values="gap_from_national_pp").reindex(
        index=STATE_ORDER, columns=["Overall"] + [f"QA{i}" for i in range(1, 8)]
    )
    benchmark_n = benchmark_table.pivot(
        index="state", columns="measure", values="rated_services"
    ).reindex(index=benchmark.index, columns=benchmark.columns)
    benchmark_labels = np.empty(benchmark.shape, dtype=object)
    for row in range(benchmark.shape[0]):
        for column in range(benchmark.shape[1]):
            benchmark_labels[row, column] = (
                f"{benchmark.iloc[row, column]:+.1f}\n"
                f"n={int(benchmark_n.iloc[row, column]):,}"
            )
    max_abs = max(2, float(np.nanmax(np.abs(benchmark.to_numpy()))))
    sns.heatmap(
        benchmark,
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=benchmark_labels,
        fmt="",
        annot_kws={"fontsize": 7.5},
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Gap from national Meeting+ rate (pp)"},
        ax=ax_heat,
    )
    ax_heat.set_title("Quality benchmark gaps", loc="left")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", rotation=0)
    ax_heat.tick_params(axis="y", rotation=0)
    fig.suptitle("Quality outcomes vary across jurisdictions and remoteness groups", x=0.08, y=0.995, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.08, 0.02, "Compositions use recognised ratings only. Benchmark cells show percentage-point gaps from the national Meeting+ rate and the rated denominator n.", color="#64748B", fontsize=9)
    _save(fig, path)


def plot_offering_quality(tables, path):
    summary = tables["offering_summary"].sort_values("below_nqs_pct_of_rated")
    state = tables["state_summary"]
    national_below = 100 * state["below_nqs_services"].sum() / state["rated_services"].sum()
    y = np.arange(len(summary))
    fig, (ax_quality, ax_coverage) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True, gridspec_kw={"wspace": 0.28})

    rates = summary["below_nqs_pct_of_rated"].to_numpy()
    ax_quality.errorbar(
        rates,
        y,
        xerr=[rates - summary["below_ci95_lower_pct"], summary["below_ci95_upper_pct"] - rates],
        fmt="o",
        color="#D1495B",
        ecolor="#94A3B8",
        capsize=4,
        markersize=8,
    )
    ax_quality.axvline(national_below, color="#475569", linestyle="--", linewidth=1.2)
    ax_quality.set_yticks(y, summary["offering"])
    ax_quality.set_xlabel("Below NQS among rated services (%)")
    ax_quality.set_title("Below-NQS rate", loc="left")
    _clean_axis(ax_quality, "x")

    ax_coverage.scatter(summary["rating_coverage_pct"], y, color="#2A9D8F", s=65)
    for row_index, row in enumerate(summary.itertuples()):
        ax_coverage.text(row.rating_coverage_pct + 0.35, row_index, f"n={row.rated_services:,}", va="center", fontsize=8, color="#475569")
    ax_coverage.set_xlim(max(0, summary["rating_coverage_pct"].min() - 5), 103)
    ax_coverage.set_xlabel("Recognised rating coverage (%)")
    ax_coverage.set_title("Rating coverage and n", loc="left")
    _clean_axis(ax_coverage, "x")
    fig.suptitle("Appendix: detailed offerings have different quality profiles", x=0.07, y=1.01, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.07, 0.01, "Error bars are 95% Wilson intervals. The dashed line is the national below-NQS rate; offerings overlap and are not independent groups.", color="#64748B", fontsize=9)
    _save(fig, path)


def plot_accessibility(tables, path):
    transport = tables["transport_by_remoteness"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.6, 8.4),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    x = np.arange(len(REMOTENESS_ORDER))
    colors = {"Bus": "#4C78A8", "Train": "#F28E2B"}
    for ax, mode in zip(axes, ["Bus", "Train"]):
        group = (
            transport[transport["transport_mode"].eq(mode)]
            .set_index("remoteness_area")
            .reindex(REMOTENESS_ORDER)
        )
        median = group["median_distance_km"].to_numpy()
        p90 = group["p90_distance_km"].to_numpy()
        ax.fill_between(
            x,
            median,
            p90,
            color=colors[mode],
            alpha=0.11,
            label="Median-to-90th-percentile range",
        )
        ax.plot(
            x,
            p90,
            color=colors[mode],
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            alpha=0.65,
            label="90th percentile",
        )
        ax.plot(
            x,
            median,
            color=colors[mode],
            marker="o",
            linewidth=2.8,
            markersize=7,
            label="Median",
        )
        for position, value in zip(x, median):
            ax.annotate(
                f"{value:,.1f} km",
                (position, value),
                xytext=(0, -15 if mode == "Bus" else 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#334155",
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(_kilometre_tick))
        ax.set_ylim(0.6, 1100)
        ax.set_ylabel("Distance (log scale)")
        ax.set_title(
            f"{mode}: typical distance and the long-access tail",
            loc="left",
            fontsize=13,
        )
        ax.grid(axis="y", which="major")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].set_xticks(
        x, [REMOTENESS_SHORT[item] for item in REMOTENESS_ORDER]
    )
    axes[1].set_xlabel("ABS 2021 Remoteness Area")
    fig.suptitle(
        "Transport distance grows rapidly with remoteness, including its long tail",
        x=0.08,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.015,
        "Lines use all services with valid distance data. The shaded range from the median to P90 exposes skew without treating distance as travel time, frequency, affordability, or actual family behaviour.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_unadjusted_quality_accessibility(tables, path):
    table = tables["accessibility_quality_quintiles"]
    scopes = ["All services", "Centre-Based Care", "Family Day Care"]
    colors = {"All services": "#111827", "Centre-Based Care": "#2A9D8F", "Family Day Care": "#E76F51"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, mode in zip(axes, ["Bus", "Train"]):
        for scope in scopes:
            group = table[table["scope"].eq(scope) & table["transport_mode"].eq(mode)].sort_values("distance_quintile")
            ax.plot(group["distance_quintile"], group["meeting_or_above_pct"], marker="o", linewidth=2, color=colors[scope], label=scope)
            if scope == "Family Day Care":
                ax.fill_between(group["distance_quintile"], group["ci95_lower_pct"], group["ci95_upper_pct"], color=colors[scope], alpha=0.12)
        ax.set_xticks(range(1, 6), ["Q1\nMost access", "Q2", "Q3", "Q4", "Q5\nLeast access"])
        ax.set_title(mode, loc="left")
        _clean_axis(ax, "y")
    axes[0].set_ylabel("Meeting NQS or above among rated services (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Appendix: unadjusted quality rates across distance quintiles", x=0.07, y=1.01, ha="left", fontsize=18, fontweight="bold")
    fig.subplots_adjust(bottom=0.17)
    fig.text(0.07, 0.02, "Quintiles use national distance cut-points. Family Day Care uncertainty is shaded; associations remain unadjusted for geography and service mix.", color="#64748B", fontsize=9)
    _save(fig, path)


def plot_adjusted_quality_accessibility(tables, path):
    predictions = tables["quality_adjusted_predictions"]
    coefficients = tables["quality_model_coefficients"]
    unadjusted = tables["accessibility_quality_quintiles"]
    colors = {"Bus": "#4C78A8", "Train": "#F28E2B"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharey=True)

    for ax, mode in zip(axes, ["Bus", "Train"]):
        prediction = predictions[predictions["transport_mode"].eq(mode)]
        raw = unadjusted[
            unadjusted["scope"].eq("All services")
            & unadjusted["transport_mode"].eq(mode)
        ].sort_values("distance_quintile")
        coefficient = coefficients[
            coefficients["transport_mode"].eq(mode)
        ].iloc[0]

        ax.fill_between(
            prediction["distance_km"],
            prediction["ci95_lower_pct"],
            prediction["ci95_upper_pct"],
            color=colors[mode],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            prediction["distance_km"],
            prediction["adjusted_meeting_or_above_pct"],
            color=colors[mode],
            linewidth=2.5,
            label="Adjusted marginal prediction",
        )
        raw_rate = raw["meeting_or_above_pct"].to_numpy()
        ax.plot(
            raw["median_distance_km"],
            raw_rate,
            marker="o",
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            markersize=6,
            color="#475569",
            alpha=0.9,
            label="Unadjusted quintile trend",
            zorder=3,
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(_kilometre_tick))
        ax.set_xlabel(f"{mode} distance (log scale)")
        ax.set_title(mode, loc="left")
        ax.text(
            0.03,
            0.05,
            "Adjusted OR per doubling\n"
            f"{coefficient['odds_ratio_per_distance_doubling']:.3f} "
            f"({coefficient['or_ci95_lower']:.3f}-"
            f"{coefficient['or_ci95_upper']:.3f})",
            transform=ax.transAxes,
            fontsize=9,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "pad": 5},
        )
        _clean_axis(ax, "y")

    axes[0].set_ylabel("Meeting NQS or above among rated services (%)")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.22))
    fig.suptitle(
        "Adjustment substantially attenuates the raw quality-accessibility gradient",
        x=0.07,
        y=1.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.01,
        "Binomial GLMs control for remoteness, state, and broad service type. Curves average predictions over the observed covariate distribution; raw-quintile uncertainty is retained in Appendix A03. Associations are not causal.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_population_adjusted_coverage(tables, boundaries, path):
    """Map Centre-Based approved places against the 0-13 population."""
    detail = tables["population_coverage_by_state_remoteness"].copy()
    national = float(
        tables["population_coverage_national"].loc[
            0, "approved_places_per_1000_children"
        ]
    )
    coverage_min = float(detail["approved_places_per_1000_children"].min())
    coverage_max = float(detail["approved_places_per_1000_children"].max())
    norm = TwoSlopeNorm(vmin=coverage_min, vcenter=national, vmax=coverage_max)
    cmap = "RdYlGn"

    geography = boundaries[
        boundaries["boundary_state_abbr"].isin(STATE_ORDER)
    ].copy()
    geography["RA_CODE21"] = geography["RA_CODE21"].astype("string")
    geography = geography.merge(
        detail,
        left_on="RA_CODE21",
        right_on="remoteness_code",
        how="left",
        validate="one_to_one",
    ).to_crs("EPSG:3577")

    fig, (ax_map, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(17, 8.3),
        gridspec_kw={"width_ratios": [1.02, 1.18], "wspace": 0.15},
    )
    geography.plot(
        column="approved_places_per_1000_children",
        cmap=cmap,
        norm=norm,
        linewidth=0.35,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label": "Approved places per 1,000 children aged 0-13",
            "shrink": 0.72,
            "pad": 0.015,
        },
        missing_kwds={"color": "#E5E7EB"},
        ax=ax_map,
    )
    geography.dissolve(by="boundary_state_abbr").boundary.plot(
        color="#334155", linewidth=0.8, ax=ax_map
    )
    ax_map.set_axis_off()
    ax_map.set_title(
        "Coverage varies within as well as between states",
        loc="left",
        fontsize=13,
    )

    matrix = detail.pivot(
        index="state",
        columns="remoteness_area",
        values="approved_places_per_1000_children",
    ).reindex(index=STATE_ORDER, columns=REMOTENESS_ORDER)
    population_matrix = detail.pivot(
        index="state",
        columns="remoteness_area",
        values="child_population_0_13",
    ).reindex(index=STATE_ORDER, columns=REMOTENESS_ORDER)
    annotations = matrix.copy().astype(object)
    for state in STATE_ORDER:
        for area in REMOTENESS_ORDER:
            value = matrix.loc[state, area]
            child_population = population_matrix.loc[state, area]
            annotations.loc[state, area] = (
                ""
                if pd.isna(value)
                else f"{value:.0f}{'*' if child_population < 5000 else ''}"
            )
    sns.heatmap(
        matrix,
        cmap=cmap,
        norm=norm,
        annot=annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar=False,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        ax=ax_heat,
    )
    ax_heat.set_xticklabels(
        [REMOTENESS_SHORT[area] for area in REMOTENESS_ORDER], rotation=25, ha="right"
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_title(
        f"State x remoteness benchmark | national = {national:.0f}",
        loc="left",
        fontsize=13,
    )

    fig.suptitle(
        "Population adjustment reveals lower approved-place intensity outside major cities",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.015,
        "Numerator: current-register Centre-Based approved places. Denominator: ABS 2021 Census usual residents aged 0-13. Red cells are below the national benchmark; * marks fewer than 5,000 children. Places are not vacancies and the dates differ.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_operations(tables, path):
    operations = tables["operations_by_remoteness"]
    effects = tables["operational_model_effects"]
    screening = tables["compound_disadvantage"]
    capacity_reference = float(
        screening["national_median_approved_places"].dropna().iloc[0]
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9.2),
        gridspec_kw={"hspace": 0.43, "wspace": 0.35},
    )
    ax_capacity, ax_hours = axes[0]
    ax_capacity_effect, ax_hours_effect = axes[1]
    y = np.arange(len(REMOTENESS_ORDER))

    centre = operations[
        operations["service_type"].eq("Centre-Based Care")
    ].set_index("remoteness_area").reindex(REMOTENESS_ORDER)
    capacity = centre["median_approved_places"].to_numpy()
    geographic_colors = ["#16324F", "#2A9D8F", "#72B7B2", "#F4A261", "#D1495B"]
    bars = ax_capacity.barh(
        y,
        capacity,
        color=geographic_colors,
        height=0.58,
    )
    for bar, value in zip(bars, capacity):
        ax_capacity.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_capacity.axvline(
        capacity_reference,
        color="#64748B",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        label=f"National median ({capacity_reference:.0f})",
    )
    ax_capacity.set_yticks(y, [REMOTENESS_SHORT[item] for item in REMOTENESS_ORDER])
    ax_capacity.invert_yaxis()
    ax_capacity.set_xlim(0, max(capacity) * 1.28)
    ax_capacity.set_xlabel("Median approved places")
    ax_capacity.set_title("Centre-Based capacity falls with remoteness", loc="left")
    ax_capacity.legend(frameon=False, loc="lower right")
    _clean_axis(ax_capacity, "x")

    colors = {"Centre-Based Care": "#2A9D8F", "Family Day Care": "#E76F51"}
    centre_hours = centre["median_annual_weekly_hours"].to_numpy()
    family = operations[
        operations["service_type"].eq("Family Day Care")
    ].set_index("remoteness_area").reindex(REMOTENESS_ORDER)
    family_hours = family["median_annual_weekly_hours"].to_numpy()
    for index, (centre_value, family_value) in enumerate(
        zip(centre_hours, family_hours)
    ):
        if np.isfinite(family_value):
            ax_hours.plot(
                [family_value, centre_value],
                [index, index],
                color="#CBD5E1",
                linewidth=3,
                zorder=1,
            )
            ax_hours.annotate(
                f"FDC n={int(family.iloc[index]['annual_hours_services']):,}",
                (family_value, index),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#9A3412",
            )
        else:
            ax_hours.text(
                73,
                index,
                "No FDC annual-hours record",
                ha="left",
                va="center",
                fontsize=8,
                color="#94A3B8",
            )
    ax_hours.scatter(
        centre_hours,
        y,
        color=colors["Centre-Based Care"],
        s=62,
        label="Centre-Based Care",
        zorder=3,
    )
    valid_family = np.isfinite(family_hours)
    ax_hours.scatter(
        family_hours[valid_family],
        y[valid_family],
        color=colors["Family Day Care"],
        s=62,
        label="Family Day Care",
        zorder=3,
    )
    ax_hours.set_yticks(y, [REMOTENESS_SHORT[item] for item in REMOTENESS_ORDER])
    ax_hours.invert_yaxis()
    ax_hours.set_xlim(20, 95)
    ax_hours.set_xlabel("Median annual-calendar hours per week")
    ax_hours.set_title("Opening-hour medians differ by type and geography", loc="left")
    ax_hours.legend(frameon=False, loc="upper right")
    _clean_axis(ax_hours, "x")

    effect_specs = [
        (ax_capacity_effect, "Centre-Based approved capacity", (-2.35, 0.45), "%"),
        (ax_hours_effect, "Annual weekly opening hours", (-0.26, 0.20), " h"),
    ]
    for ax, outcome, limits, suffix in effect_specs:
        group = effects[effects["outcome"].eq(outcome)].set_index(
            "transport_mode"
        ).reindex(["Bus", "Train"])
        estimate = group["adjusted_effect_per_distance_doubling"].to_numpy()
        lower = group["ci95_lower"].to_numpy()
        upper = group["ci95_upper"].to_numpy()
        excludes_zero = (lower > 0) | (upper < 0)
        effect_colors = np.where(excludes_zero, "#2A9D8F", "#94A3B8")
        ax.barh(
            [0, 1],
            estimate,
            color=effect_colors,
            height=0.42,
        )
        for row_index, (value, low, high) in enumerate(
            zip(estimate, lower, upper)
        ):
            ax.text(
                0.98,
                row_index,
                f"{value:+.2f}{suffix}  (95% CI {low:+.2f} to {high:+.2f})",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.5,
                color="#334155",
            )
        ax.axvline(0, color="#64748B", linestyle=(0, (4, 3)), linewidth=1)
        ax.set_yticks([0, 1], ["Bus", "Train"])
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.set_xlabel("Adjusted effect per distance doubling")
        effect_title = (
            "Adjusted capacity effect"
            if outcome == "Centre-Based approved capacity"
            else "Adjusted opening-hours effect"
        )
        ax.set_title(effect_title, loc="left")
        _clean_axis(ax, "x")

    fig.suptitle(
        "Lower remote capacity is visible; opening-hours evidence is less consistent",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Capacity is Centre-Based only. Family Day Care hour medians have very small remote samples. Models control remoteness and state; hours also control broad type. Robust confidence intervals are printed beside each effect.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_approval_cohorts(tables, path):
    approval = tables["approval_year_by_service_type"]
    pivot = approval.pivot(
        index="approval_year", columns="service_type", values="current_services"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    bottom = np.zeros(len(pivot))
    for service_type, color in [
        ("Centre-Based Care", "#457B9D"),
        ("Family Day Care", "#E76F51"),
    ]:
        values = pivot.get(service_type, pd.Series(0, index=pivot.index)).to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, color=color, label=service_type)
        bottom += values
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Original approval year")
    ax.set_ylabel("Services in the current register")
    ax.set_title("Appendix: approval-year composition of currently registered services", loc="left", fontsize=17, pad=12)
    ax.legend(frameon=False)
    _clean_axis(ax, "y")
    fig.text(0.125, 0.02, "This is not historical sector growth: services that closed before the current register snapshot are absent.", color="#64748B", fontsize=9)
    _save(fig, path)


def plot_synthesis(tables, path):
    table = tables["compound_disadvantage"]
    plot_data = table[
        table["eligible_for_comparison"] & table["enhanced_screening_flag"]
    ].copy()
    quality_reference = float(plot_data["national_below_nqs_pct"].iloc[0])
    transport_reference = float(
        plot_data["national_median_transport_km"].iloc[0]
    )
    capacity_reference = float(
        plot_data["national_median_approved_places"].iloc[0]
    )
    coverage_reference = float(
        plot_data["national_approved_places_per_1000_children"].iloc[0]
    )
    plot_data["quality_gap_pp"] = (
        plot_data["below_nqs_pct_of_rated"] - quality_reference
    )
    plot_data["transport_multiple"] = (
        plot_data["median_nearest_transport_km"] / transport_reference
    )
    plot_data["transport_doublings"] = np.log2(plot_data["transport_multiple"])
    plot_data["capacity_shortfall_pct"] = (
        1 - plot_data["median_approved_places"] / capacity_reference
    ) * 100
    plot_data["coverage_shortfall_pct"] = (
        1
        - plot_data["approved_places_per_1000_children"]
        / coverage_reference
    ) * 100
    plot_data = plot_data.sort_values("quality_gap_pp", ascending=False).reset_index(
        drop=True
    )

    y = np.arange(len(plot_data))
    labels = [
        f"{row.state} | {REMOTENESS_SHORT[row.remoteness_area]}  (rated n={int(row.rated_services)})"
        for row in plot_data.itertuples()
    ]
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19, 8.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.0, 1.0], "wspace": 0.12},
    )
    specs = [
        (
            axes[0],
            plot_data["quality_gap_pp"].to_numpy(),
            "#D1495B",
            "Quality gap",
            "Below-NQS gap (pp)",
        ),
        (
            axes[1],
            plot_data["transport_doublings"].to_numpy(),
            "#F4A261",
            "Transport access gap",
            "Distance relative to national median",
        ),
        (
            axes[2],
            plot_data["capacity_shortfall_pct"].to_numpy(),
            "#2A9D8F",
            "Facility-size gap",
            "Median capacity shortfall (%)",
        ),
        (
            axes[3],
            plot_data["coverage_shortfall_pct"].to_numpy(),
            "#7C3AED",
            "Population coverage gap",
            "Places/1,000 shortfall (%)",
        ),
    ]
    for ax, values, color, title, xlabel in specs:
        ax.barh(y, values, color=color, height=0.62, alpha=0.9)
        ax.axvline(0, color="#64748B", linewidth=1)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.55)
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(values) * 1.24)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=8.5, labelleft=True)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    axes[3].tick_params(axis="y", labelleft=False)

    for row_index, row in plot_data.iterrows():
        axes[0].text(
            row["quality_gap_pp"] + 0.3,
            row_index,
            f"+{row['quality_gap_pp']:.1f}",
            va="center",
            fontsize=8,
            color="#7F1D1D",
        )
        axes[1].text(
            row["transport_doublings"] + 0.08,
            row_index,
            f"{row['transport_multiple']:.1f}x",
            va="center",
            fontsize=8,
            color="#9A3412",
        )
        axes[2].text(
            row["capacity_shortfall_pct"] + 0.8,
            row_index,
            f"{row['capacity_shortfall_pct']:.0f}%",
            va="center",
            fontsize=8,
            color="#115E59",
        )
        axes[3].text(
            row["coverage_shortfall_pct"] + 0.8,
            row_index,
            f"{row['approved_places_per_1000_children']:.0f}",
            va="center",
            fontsize=8,
            color="#5B21B6",
        )

    transport_ticks = np.arange(
        0, np.floor(plot_data["transport_doublings"].max()) + 1, 1
    )
    axes[1].set_xticks(
        transport_ticks,
        [f"{2 ** int(value):g}x" for value in transport_ticks],
    )
    fig.suptitle(
        f"{len(plot_data)} geographic groups cross all four screening benchmarks",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.015,
        "Centre-Based Care only. Every row has above-benchmark Below-NQS and transport distance, below-benchmark median facility size, and below-benchmark approved places per 1,000 children. Coverage labels show places per 1,000; no weighted score is created.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def create_visualisations(data, tables, boundaries, output_dir):
    """Create static figures that complement, rather than duplicate, interaction."""
    _configure_style()
    paths = output_paths(output_dir)
    figures = Path(paths["figures"])
    presentation = Path(paths["presentation_figures"])
    appendix = Path(paths["appendix_figures"])
    for directory in [figures, presentation, appendix]:
        for stale_figure in directory.glob("*.png"):
            stale_figure.unlink()

    plot_spatial_landscape(
        data, boundaries, presentation / "01_spatial_landscape.png"
    )
    plot_offering_mix_by_remoteness(
        tables, presentation / "02_service_mix_confounder.png"
    )
    plot_quality(tables, presentation / "03_quality_geography.png")
    plot_accessibility(tables, presentation / "04_accessibility_geography.png")
    plot_population_adjusted_coverage(
        tables,
        boundaries,
        presentation / "04b_population_adjusted_coverage.png",
    )
    plot_adjusted_quality_accessibility(
        tables, presentation / "05_adjusted_quality_accessibility.png"
    )
    plot_operations(tables, presentation / "06_operational_geography.png")
    plot_synthesis(tables, presentation / "07_multidimensional_screening.png")

    plot_service_offering_portfolio(
        tables, appendix / "A01_service_offering_portfolio.png"
    )
    plot_offering_quality(tables, appendix / "A02_quality_by_offering.png")
    plot_unadjusted_quality_accessibility(
        tables, appendix / "A03_unadjusted_quality_accessibility.png"
    )
    plot_approval_cohorts(tables, appendix / "A04_approval_cohorts.png")
