"""Generate exploratory figures that document and motivate the final analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    RATING_GROUP_COLORS,
    RATING_GROUP_ORDER,
    REMOTENESS_ORDER,
    STATE_ORDER,
    output_paths,
)


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
            "axes.titlesize": 13,
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


def plot_data_readiness(audit_tables, path):
    """Show variable completeness and analysis-specific sample sizes."""
    missing = audit_tables["missingness_summary"].copy()
    sample_flow = audit_tables["analysis_sample_flow"].copy()
    labels = {
        "OverallRating": "Overall rating",
        "rating_date": "Rating date",
        "NumberOfApprovedPlaces": "Approved places",
        "annual_weekly_operating_hours": "Annual hours",
        "term_weekly_operating_hours": "School-term hours",
        "holiday_weekly_operating_hours": "Holiday hours",
        "remoteness_area": "ABS remoteness",
        "DistanceToBusStation_km": "Bus distance",
        "DistanceToTrainStation_km": "Train distance",
        "approval_date": "Approval date",
        "State": "State",
        "longitude": "Longitude",
        "latitude": "Latitude",
    }
    missing["label"] = missing["variable"].map(labels).fillna(missing["variable"])
    missing = missing.sort_values("missing_pct", ascending=True)

    fig, (ax_missing, ax_sample) = plt.subplots(
        1, 2, figsize=(15.5, 7.2), gridspec_kw={"wspace": 0.38}
    )
    colors = np.where(missing["missing_pct"].gt(20), "#D1495B", "#457B9D")
    bars = ax_missing.barh(
        missing["label"], missing["missing_pct"], color=colors, height=0.65
    )
    for bar, row in zip(bars, missing.itertuples()):
        ax_missing.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{row.missing_pct:.1f}%",
            va="center",
            fontsize=8.5,
        )
    ax_missing.set_xlim(0, max(80, missing["missing_pct"].max() + 8))
    ax_missing.set_xlabel("Missing records (%)")
    ax_missing.set_title("Completeness differs by analytical variable", loc="left")
    _clean_axis(ax_missing, "x")
    ax_missing.spines["left"].set_visible(False)

    sample_flow = sample_flow.iloc[::-1].copy()
    sample_colors = ["#16324F", "#2A9D8F", "#457B9D", "#72B7B2", "#F4A261", "#D1495B"]
    sample_bars = ax_sample.barh(
        sample_flow["analysis_sample"],
        sample_flow["share_of_source_pct"],
        color=sample_colors[: len(sample_flow)],
        height=0.62,
    )
    for bar, row in zip(sample_bars, sample_flow.itertuples()):
        ax_sample.text(
            min(bar.get_width() + 1, 98.5),
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.services):,} ({row.share_of_source_pct:.1f}%)",
            va="center",
            ha="right" if bar.get_width() > 94 else "left",
            color="white" if bar.get_width() > 94 else "#334155",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_sample.set_xlim(0, 100)
    ax_sample.set_xlabel("Share of source services (%)")
    ax_sample.set_title("Samples are defined separately for each analysis", loc="left")
    _clean_axis(ax_sample, "x")
    ax_sample.spines["left"].set_visible(False)

    fig.suptitle(
        "EDA 1 | Data readiness and analysis-specific denominators",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Rows are not globally deleted for one missing field. Red completeness bars exceed 20% missing; each analysis uses only the variables it requires.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_univariate_profile(data, path):
    """Explore categorical and numeric distributions before modelling."""
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 10), gridspec_kw={"hspace": 0.42, "wspace": 0.30}
    )
    ax_rating, ax_transport, ax_capacity, ax_hours = axes.ravel()

    rating_counts = (
        data["rating_group"].value_counts().reindex(RATING_GROUP_ORDER, fill_value=0)
    )
    rating_pct = rating_counts.div(len(data)).mul(100)
    rating_bars = ax_rating.barh(
        rating_counts.index,
        rating_pct,
        color=[RATING_GROUP_COLORS[item] for item in rating_counts.index],
        height=0.62,
    )
    for bar, count, pct in zip(rating_bars, rating_counts, rating_pct):
        ax_rating.text(
            bar.get_width() + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    ax_rating.set_xlim(0, max(rating_pct) * 1.28)
    ax_rating.set_xlabel("Share of all services (%)")
    ax_rating.set_title("Overall-rating status", loc="left")
    _clean_axis(ax_rating, "x")
    ax_rating.spines["left"].set_visible(False)

    transport = data["nearest_transport_distance_km"].dropna().clip(lower=0.03)
    transport_bins = np.geomspace(0.03, transport.max(), 48)
    ax_transport.hist(transport, bins=transport_bins, color="#457B9D", alpha=0.86)
    ax_transport.set_xscale("log")
    transport_median = transport.median()
    transport_p90 = transport.quantile(0.9)
    ax_transport.axvline(transport_median, color="#16324F", linewidth=2)
    ax_transport.axvline(
        transport_p90, color="#D1495B", linewidth=1.7, linestyle=(0, (4, 3))
    )
    ax_transport.text(
        0.03,
        0.92,
        f"Median {transport_median:.1f} km\nP90 {transport_p90:.1f} km",
        transform=ax_transport.transAxes,
        va="top",
        color="#334155",
    )
    ax_transport.set_xlabel("Nearest listed transport distance (km, log scale)")
    ax_transport.set_ylabel("Services")
    ax_transport.set_title("Transport distance is strongly right-skewed", loc="left")
    _clean_axis(ax_transport, "y")

    capacity = data.loc[
        data["ServiceType"].eq("Centre-Based Care")
        & data["NumberOfApprovedPlaces"].gt(0),
        "NumberOfApprovedPlaces",
    ]
    capacity_p99 = capacity.quantile(0.99)
    ax_capacity.hist(
        capacity[capacity.le(capacity_p99)],
        bins=35,
        color="#2A9D8F",
        alpha=0.86,
    )
    ax_capacity.axvline(capacity.median(), color="#16324F", linewidth=2)
    ax_capacity.text(
        0.97,
        0.92,
        f"Median {capacity.median():.0f} places\n{capacity.gt(capacity_p99).sum():,} records above P99 retained",
        transform=ax_capacity.transAxes,
        ha="right",
        va="top",
        color="#334155",
    )
    ax_capacity.set_xlabel("Approved places (display to P99)")
    ax_capacity.set_ylabel("Centre-Based services")
    ax_capacity.set_title("Centre-Based capacity has a plausible upper tail", loc="left")
    _clean_axis(ax_capacity, "y")

    hour_colors = {"Centre-Based Care": "#2A9D8F", "Family Day Care": "#E76F51"}
    for service_type in ["Centre-Based Care", "Family Day Care"]:
        values = data.loc[
            data["ServiceType"].eq(service_type) & data["annual_hours_valid"],
            "annual_weekly_operating_hours",
        ].dropna()
        upper = values.quantile(0.99)
        ax_hours.hist(
            values[values.le(upper)],
            bins=34,
            density=True,
            histtype="step",
            linewidth=2.1,
            color=hour_colors[service_type],
            label=f"{service_type} (median {values.median():.1f} h)",
        )
    ax_hours.set_xlabel("Annual-calendar operating hours per week (display to P99)")
    ax_hours.set_ylabel("Density")
    ax_hours.set_title("Operating-hours structure differs by broad type", loc="left")
    ax_hours.legend(frameon=False, fontsize=8.5)
    _clean_axis(ax_hours, "y")

    fig.suptitle(
        "EDA 2 | Outcome and operational distributions",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Display limits improve readability only; valid upper-tail observations remain in the dataset and in model preparation. Capacity is restricted to Centre-Based Care.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_geographic_sample_structure(data, path):
    """Reveal the state-remoteness structure and sparse comparison groups."""
    selected = data[
        data["State"].isin(STATE_ORDER)
        & data["remoteness_area"].isin(REMOTENESS_ORDER)
    ].copy()
    counts = pd.crosstab(selected["State"], selected["remoteness_area"]).reindex(
        index=STATE_ORDER, columns=REMOTENESS_ORDER, fill_value=0
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    annotations = np.empty(counts.shape, dtype=object)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count = int(counts.iloc[row, column])
            annotations[row, column] = (
                f"{shares.iloc[row, column]:.1f}%\nn={count:,}" if count else "-"
            )

    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    sns.heatmap(
        shares,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 8.5},
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Share of services within state (%)"},
        ax=ax,
    )
    ax.set_xticklabels(
        [REMOTENESS_SHORT[item] for item in REMOTENESS_ORDER], rotation=0
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "EDA 3 | State samples have very different remoteness structures",
        loc="left",
        pad=16,
        fontsize=18,
    )
    fig.text(
        0.125,
        0.01,
        "Percentages are calculated within each state; counts identify sparse Remote and Very Remote cells that require uncertainty intervals or minimum-n rules.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def _distance_tick(value, _position):
    distance = 10**value - 0.1
    if distance >= 1000:
        return f"{distance / 1000:.0f}k"
    if distance >= 10:
        return f"{distance:.0f}"
    return f"{distance:.1f}"


def _relationship_panel(ax, frame, outcome, title, color, y_label):
    selected = frame[["nearest_transport_distance_km", outcome]].dropna().copy()
    selected = selected[selected["nearest_transport_distance_km"].ge(0)]
    selected["log_distance"] = np.log10(
        selected["nearest_transport_distance_km"] + 0.1
    )
    display_upper = selected[outcome].quantile(0.99)
    hexbin = ax.hexbin(
        selected["log_distance"],
        selected[outcome],
        gridsize=38,
        bins="log",
        mincnt=1,
        cmap="Blues",
        linewidths=0,
    )
    selected["distance_decile"] = pd.qcut(
        selected["log_distance"], 10, labels=False, duplicates="drop"
    )
    trend = selected.groupby("distance_decile", observed=True).agg(
        log_distance=("log_distance", "median"),
        outcome=(outcome, "median"),
    )
    ax.plot(
        trend["log_distance"],
        trend["outcome"],
        color=color,
        marker="o",
        linewidth=2.3,
        markersize=5,
        label="Median by distance decile",
    )
    tick_distances = np.array([0.1, 1, 10, 100, 1000])
    ax.set_xticks(np.log10(tick_distances + 0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(_distance_tick))
    ax.set_ylim(0, display_upper * 1.08)
    ax.set_xlabel("Nearest listed transport distance (km, log display)")
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    return hexbin, int(selected[outcome].gt(display_upper).sum())


def plot_raw_relationship_diagnostics(data, path):
    """Inspect raw service-level operational relationships before adjustment."""
    capacity_data = data[
        data["ServiceType"].eq("Centre-Based Care")
        & data["NumberOfApprovedPlaces"].gt(0)
    ]
    hours_data = data[data["annual_hours_valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), gridspec_kw={"wspace": 0.28})
    capacity_hex, capacity_tail = _relationship_panel(
        axes[0],
        capacity_data,
        "NumberOfApprovedPlaces",
        "Centre-Based capacity vs transport distance",
        "#2A9D8F",
        "Approved places (display to P99)",
    )
    hours_hex, hours_tail = _relationship_panel(
        axes[1],
        hours_data,
        "annual_weekly_operating_hours",
        "Annual opening hours vs transport distance",
        "#E76F51",
        "Annual operating hours per week (display to P99)",
    )
    for ax, hexbin in zip(axes, [capacity_hex, hours_hex]):
        colorbar = fig.colorbar(hexbin, ax=ax, fraction=0.035, pad=0.015)
        colorbar.set_label("Log cell count")
    axes[0].text(
        0.02,
        0.94,
        f"{capacity_tail:,} values above display P99 retained",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        color="#64748B",
    )
    axes[1].text(
        0.02,
        0.94,
        f"{hours_tail:,} values above display P99 retained",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        color="#64748B",
    )
    fig.suptitle(
        "EDA 4 | Raw operational relationships require geographic adjustment",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Hexagons reduce overplotting; trend lines show unadjusted medians only. These patterns motivate models controlling for remoteness, state, and service type and must not be interpreted causally.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def plot_descriptive_statistics_table(table, path):
    """Render the key numeric EDA summary as a presentation-ready table."""
    display = table.copy()
    display_columns = [
        ("measure", "Measure"),
        ("scope", "Analytical scope"),
        ("valid_n", "Valid n"),
        ("missing_pct", "Missing %"),
        ("mean", "Mean"),
        ("std_dev", "SD"),
        ("minimum", "Min"),
        ("q1", "Q1"),
        ("median", "Median"),
        ("q3", "Q3"),
        ("p90", "P90"),
        ("maximum", "Max"),
    ]
    formatted = []
    for row in display.itertuples(index=False):
        record = row._asdict()
        formatted.append(
            [
                record[column]
                if column in {"measure", "scope"}
                else f"{int(record[column]):,}"
                if column == "valid_n"
                else f"{record[column]:,.1f}"
                for column, _label in display_columns
            ]
        )

    fig, ax = plt.subplots(figsize=(17.2, 5.4))
    ax.axis("off")
    column_widths = [0.19, 0.13, 0.07, 0.075] + [0.065] * 8
    summary_table = ax.table(
        cellText=formatted,
        colLabels=[label for _column, label in display_columns],
        colWidths=column_widths,
        cellLoc="right",
        colLoc="right",
        bbox=[0.02, 0.16, 0.96, 0.68],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    summary_table.scale(1, 1.55)
    for (row, column), cell in summary_table.get_celld().items():
        cell.set_edgecolor("#D9E2E7")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#16324F")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F7FAFC" if row % 2 else "white")
        if column in {0, 1}:
            cell.get_text().set_ha("left")

    fig.suptitle(
        "EDA 5 | Key descriptive statistics and valid denominators",
        x=0.03,
        y=0.95,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.07,
        "Distances and approved places are strongly right-skewed, so medians, quartiles and P90 are retained alongside means. Approved places use Centre-Based Care only; annual hours use annual-calendar records only.",
        color="#64748B",
        fontsize=9,
    )
    _save(fig, path)


def create_eda_visualisations(data, tables, audit_tables, output_dir):
    """Create the dedicated EDA figure set in a reproducible output folder."""
    _configure_style()
    eda_dir = Path(output_paths(output_dir)["eda_figures"])
    for stale in eda_dir.glob("*.png"):
        stale.unlink()
    plot_data_readiness(audit_tables, eda_dir / "E01_data_readiness.png")
    plot_univariate_profile(data, eda_dir / "E02_univariate_profile.png")
    plot_geographic_sample_structure(
        data, eda_dir / "E03_geographic_sample_structure.png"
    )
    plot_raw_relationship_diagnostics(
        data, eda_dir / "E04_raw_relationship_diagnostics.png"
    )
    plot_descriptive_statistics_table(
        tables["eda_descriptive_statistics"],
        eda_dir / "E05_descriptive_statistics_table.png",
    )
