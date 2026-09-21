"""Create auditable summary tables and a concise findings report."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DETAILED_SERVICE_OFFERINGS,
    HIGH_RATINGS,
    LOW_RATINGS,
    OFFERING_ORDER,
    QUALITY_AREA_COLUMNS,
    QUALITY_AREA_LABELS,
    QUALITY_AREA_NAMES,
    RATING_ORDER,
    REMOTENESS_ORDER,
    STATE_ORDER,
    TRANSPORT_BAND_ORDER,
    output_paths,
)
from .models import build_model_outputs
from .population import (
    build_population_coverage_tables,
    enrich_screening_with_population,
)


def _percent(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return 100.0 * numerator / denominator


def _wilson_interval(successes, total, z=1.96):
    """Return a 95% Wilson interval as percentages."""
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + (z * z / total)
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return 100 * (centre - margin), 100 * (centre + margin)


def build_eda_descriptive_statistics(data):
    """Summarise key numeric variables with explicit analytical scopes."""
    centre_mask = data["ServiceType"].eq("Centre-Based Care")
    specs = [
        (
            "Bus distance (km)",
            "All services",
            data["DistanceToBusStation_km"],
            len(data),
        ),
        (
            "Train distance (km)",
            "All services",
            data["DistanceToTrainStation_km"],
            len(data),
        ),
        (
            "Nearest listed transport (km)",
            "All services",
            data["nearest_transport_distance_km"],
            len(data),
        ),
        (
            "Approved places",
            "Centre-Based Care",
            data.loc[centre_mask, "NumberOfApprovedPlaces"],
            int(centre_mask.sum()),
        ),
        (
            "Annual operating hours/week",
            "All services",
            data["annual_weekly_operating_hours"],
            len(data),
        ),
    ]
    records = []
    for measure, scope, values, scope_n in specs:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        records.append(
            {
                "measure": measure,
                "scope": scope,
                "scope_n": scope_n,
                "valid_n": len(numeric),
                "missing_n": scope_n - len(numeric),
                "missing_pct": _percent(scope_n - len(numeric), scope_n),
                "mean": numeric.mean(),
                "std_dev": numeric.std(),
                "minimum": numeric.min(),
                "q1": numeric.quantile(0.25),
                "median": numeric.median(),
                "q3": numeric.quantile(0.75),
                "p90": numeric.quantile(0.90),
                "maximum": numeric.max(),
            }
        )
    return pd.DataFrame(records)


def build_data_quality_summary(data):
    checks = [
        ("Rows", len(data)),
        ("Unique service approval numbers", data["ServiceApprovalNumber"].nunique()),
        ("Duplicate service approval numbers", data["ServiceApprovalNumber"].duplicated().sum()),
        ("Missing state", data["State"].isna().sum()),
        ("Missing or unrecognised overall rating", (~data["is_rated"]).sum()),
        ("Missing approved places", data["NumberOfApprovedPlaces"].isna().sum()),
        ("Missing approval date", data["approval_date"].isna().sum()),
        ("Missing rating date", data["rating_date"].isna().sum()),
        ("Invalid or unparsed coordinate", (~data["coordinate_valid"]).sum()),
        ("Negative train distance", data["DistanceToTrainStation_km"].lt(0).sum()),
        ("Negative bus distance", data["DistanceToBusStation_km"].lt(0).sum()),
        ("Train distance over 500 km", data["DistanceToTrainStation_km"].gt(500).sum()),
        ("Bus distance over 500 km", data["DistanceToBusStation_km"].gt(500).sum()),
    ]
    result = pd.DataFrame(checks, columns=["metric", "count"])
    result["share_of_rows_pct"] = result["count"].div(len(data)).mul(100)
    return result


def build_overall_rating_summary(data):
    labels = RATING_ORDER + ["No current rating"]
    counts = data["OverallRating"].where(data["is_rated"], "No current rating").value_counts()
    rated_total = int(data["is_rated"].sum())
    records = []
    for rating in labels:
        count = int(counts.get(rating, 0))
        records.append(
            {
                "rating": rating,
                "service_count": count,
                "share_all_services_pct": _percent(count, len(data)),
                "share_rated_services_pct": (
                    _percent(count, rated_total) if rating != "No current rating" else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def build_state_summary(data):
    records = []
    for state, group in data.dropna(subset=["State"]).groupby("State", observed=True):
        rated = group[group["is_rated"]]
        below = rated[rated["is_below_nqs"]]
        above = rated[rated["is_above_nqs"]]
        centre_based = group[group["ServiceType"].eq("Centre-Based Care")]
        records.append(
            {
                "state": state,
                "services": len(group),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": len(below),
                "below_nqs_pct_of_rated": _percent(len(below), len(rated)),
                "above_nqs_services": len(above),
                "above_nqs_pct_of_rated": _percent(len(above), len(rated)),
                "no_current_rating_services": len(group) - len(rated),
                "no_current_rating_pct": _percent(len(group) - len(rated), len(group)),
                "median_centre_based_places": centre_based["NumberOfApprovedPlaces"].median(),
                "median_train_distance_km": group["DistanceToTrainStation_km"].median(),
                "median_bus_distance_km": group["DistanceToBusStation_km"].median(),
                "valid_coordinate_pct": _percent(group["coordinate_valid"].sum(), len(group)),
            }
        )
    result = pd.DataFrame(records)
    result["state_order"] = result["state"].map({s: i for i, s in enumerate(STATE_ORDER)})
    return result.sort_values(["state_order", "state"]).drop(columns="state_order").reset_index(drop=True)


def build_quality_area_summary(data):
    records = []
    for column in QUALITY_AREA_COLUMNS:
        ratings = data[column]
        recognised = ratings.isin(RATING_ORDER)
        rated_count = int(recognised.sum())
        below_count = int(ratings.isin(LOW_RATINGS).sum())
        above_count = int(ratings.isin(HIGH_RATINGS).sum())
        label = QUALITY_AREA_LABELS[column]
        records.append(
            {
                "quality_area": label,
                "quality_area_name": QUALITY_AREA_NAMES[label],
                "rated_services": rated_count,
                "below_nqs_services": below_count,
                "below_nqs_pct_of_rated": _percent(below_count, rated_count),
                "above_nqs_services": above_count,
                "above_nqs_pct_of_rated": _percent(above_count, rated_count),
                "missing_or_unrecognised": int((~recognised).sum()),
            }
        )
    return pd.DataFrame(records)


def build_quality_area_by_state(data):
    records = []
    valid_states = data.dropna(subset=["State"])
    for state, group in valid_states.groupby("State", observed=True):
        for column in QUALITY_AREA_COLUMNS:
            ratings = group[column]
            recognised = ratings.isin(RATING_ORDER)
            rated_count = int(recognised.sum())
            below_count = int(ratings.isin(LOW_RATINGS).sum())
            records.append(
                {
                    "state": state,
                    "quality_area": QUALITY_AREA_LABELS[column],
                    "rated_services": rated_count,
                    "below_nqs_services": below_count,
                    "below_nqs_pct_of_rated": _percent(below_count, rated_count),
                }
            )
    return pd.DataFrame(records)


def build_remoteness_summary(data):
    records = []
    for area in REMOTENESS_ORDER:
        group = data[data["remoteness_area"].eq(area)]
        rated = group[group["is_rated"]]
        below = rated[rated["is_below_nqs"]]
        above = rated[rated["is_above_nqs"]]
        centre_based = group[group["ServiceType"].eq("Centre-Based Care")]
        records.append(
            {
                "remoteness_area": area,
                "services": len(group),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": len(below),
                "below_nqs_pct_of_rated": _percent(len(below), len(rated)),
                "above_nqs_services": len(above),
                "above_nqs_pct_of_rated": _percent(len(above), len(rated)),
                "no_current_rating_services": len(group) - len(rated),
                "no_current_rating_pct": _percent(len(group) - len(rated), len(group)),
                "median_centre_based_places": centre_based[
                    "NumberOfApprovedPlaces"
                ].median(),
                "median_nearest_transport_km": group[
                    "nearest_transport_distance_km"
                ].median(),
            }
        )
    return pd.DataFrame(records)


def build_remoteness_rating_composition(data):
    records = []
    for area in REMOTENESS_ORDER:
        group = data[data["remoteness_area"].eq(area)]
        rated = group[group["is_rated"]]
        counts = rated["OverallRating"].value_counts()
        for rating in RATING_ORDER:
            count = int(counts.get(rating, 0))
            records.append(
                {
                    "remoteness_area": area,
                    "rating": rating,
                    "services": count,
                    "share_of_rated_pct": _percent(count, len(rated)),
                }
            )
    return pd.DataFrame(records)


def build_quality_area_by_remoteness(data):
    records = []
    for area in REMOTENESS_ORDER:
        group = data[data["remoteness_area"].eq(area)]
        for column in QUALITY_AREA_COLUMNS:
            ratings = group[column]
            recognised = ratings.isin(RATING_ORDER)
            rated_count = int(recognised.sum())
            below_count = int(ratings.isin(LOW_RATINGS).sum())
            records.append(
                {
                    "remoteness_area": area,
                    "quality_area": QUALITY_AREA_LABELS[column],
                    "rated_services": rated_count,
                    "below_nqs_services": below_count,
                    "below_nqs_pct_of_rated": _percent(below_count, rated_count),
                }
            )
    return pd.DataFrame(records)


def build_remoteness_service_type_summary(data):
    records = []
    classified = data[data["remoteness_area"].isin(REMOTENESS_ORDER)]
    grouped = classified.groupby(
        ["remoteness_area", "ServiceType"], dropna=False, observed=True
    )
    for (area, service_type), group in grouped:
        rated = group[group["is_rated"]]
        below = rated[rated["is_below_nqs"]]
        above = rated[rated["is_above_nqs"]]
        records.append(
            {
                "remoteness_area": area,
                "service_type": service_type if pd.notna(service_type) else "Missing",
                "services": len(group),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": len(below),
                "below_nqs_pct_of_rated": _percent(len(below), len(rated)),
                "above_nqs_services": len(above),
                "above_nqs_pct_of_rated": _percent(len(above), len(rated)),
            }
        )
    result = pd.DataFrame(records)
    result["remoteness_order"] = result["remoteness_area"].map(
        {name: position for position, name in enumerate(REMOTENESS_ORDER)}
    )
    return result.sort_values(["remoteness_order", "service_type"]).drop(
        columns="remoteness_order"
    ).reset_index(drop=True)


def build_state_remoteness_summary(data):
    records = []
    classified = data.dropna(subset=["State", "remoteness_area"])
    for (state, area), group in classified.groupby(
        ["State", "remoteness_area"], observed=True
    ):
        rated = group[group["is_rated"]]
        below = rated[rated["is_below_nqs"]]
        records.append(
            {
                "state": state,
                "remoteness_area": area,
                "services": len(group),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": len(below),
                "below_nqs_pct_of_rated": _percent(len(below), len(rated)),
            }
        )
    return pd.DataFrame(records)


def build_spatial_join_diagnostics(data):
    matched = data["remoteness_area"].notna()
    comparable = data["state_boundary_consistent"].notna()
    inconsistent = data["state_boundary_consistent"].eq(False).fillna(False)
    checks = [
        ("Rows", len(data)),
        ("Valid coordinates", int(data["coordinate_valid"].sum())),
        ("Matched to ABS remoteness area", int(matched.sum())),
        (
            "Valid coordinate outside RA boundary",
            int(data["remoteness_match_status"].eq("Valid coordinate outside RA boundary").sum()),
        ),
        ("Invalid coordinates", int((~data["coordinate_valid"]).sum())),
        ("Boundary state comparison available", int(comparable.sum())),
        ("Source state differs from boundary state", int(inconsistent.sum())),
    ]
    result = pd.DataFrame(checks, columns=["metric", "count"])
    result["share_of_rows_pct"] = result["count"].div(len(data)).mul(100)
    return result


def build_service_type_summary(data):
    records = []
    for service_type, group in data.groupby("ServiceType", dropna=False, observed=True):
        rated = group[group["is_rated"]]
        below = rated[rated["is_below_nqs"]]
        above = rated[rated["is_above_nqs"]]
        records.append(
            {
                "service_type": service_type if pd.notna(service_type) else "Missing",
                "services": len(group),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": len(below),
                "below_nqs_pct_of_rated": _percent(len(below), len(rated)),
                "above_nqs_services": len(above),
                "above_nqs_pct_of_rated": _percent(len(above), len(rated)),
                "median_approved_places": group["NumberOfApprovedPlaces"].median(),
            }
        )
    return pd.DataFrame(records).sort_values("services", ascending=False).reset_index(drop=True)


def build_service_type_quality_profile(data):
    records = []
    for service_type, group in data.groupby("ServiceType", dropna=False, observed=True):
        for column in QUALITY_AREA_COLUMNS:
            ratings = group[column]
            recognised = ratings.isin(RATING_ORDER)
            rated_count = int(recognised.sum())
            below_count = int(ratings.isin(LOW_RATINGS).sum())
            records.append(
                {
                    "service_type": service_type if pd.notna(service_type) else "Missing",
                    "quality_area": QUALITY_AREA_LABELS[column],
                    "rated_services": rated_count,
                    "below_nqs_services": below_count,
                    "below_nqs_pct_of_rated": _percent(below_count, rated_count),
                }
            )
    return pd.DataFrame(records)


def build_transport_summary(data):
    records = []
    for band in TRANSPORT_BAND_ORDER:
        group = data[data["transport_band"].eq(band)]
        rated = group[group["is_rated"]]
        below_count = int(rated["is_below_nqs"].sum())
        lower, upper = _wilson_interval(below_count, len(rated))
        records.append(
            {
                "transport_band": band,
                "services": len(group),
                "rated_services": len(rated),
                "below_nqs_services": below_count,
                "below_nqs_pct_of_rated": _percent(below_count, len(rated)),
                "ci95_lower_pct": lower,
                "ci95_upper_pct": upper,
            }
        )
    return pd.DataFrame(records)


def build_capacity_quality_summary(data):
    centre_based = data[
        data["ServiceType"].eq("Centre-Based Care") & data["is_rated"]
    ].dropna(subset=["NumberOfApprovedPlaces"])
    records = []
    for group_name in ["Below NQS", "Meets NQS", "Above NQS"]:
        capacity = centre_based.loc[
            centre_based["rating_group"].eq(group_name), "NumberOfApprovedPlaces"
        ]
        records.append(
            {
                "quality_status": group_name,
                "services": len(capacity),
                "mean_approved_places": capacity.mean(),
                "median_approved_places": capacity.median(),
                "q1_approved_places": capacity.quantile(0.25),
                "q3_approved_places": capacity.quantile(0.75),
            }
        )
    return pd.DataFrame(records)


def build_approval_year_summary(data):
    valid = data.dropna(subset=["approval_year"]).copy()
    valid["approval_year"] = valid["approval_year"].astype(int)
    result = valid.groupby("approval_year").size().rename("services_approved").reset_index()
    all_years = pd.DataFrame(
        {"approval_year": range(result["approval_year"].min(), result["approval_year"].max() + 1)}
    )
    return all_years.merge(result, how="left", on="approval_year").fillna({"services_approved": 0})


def build_state_service_priority(data):
    records = []
    grouped = data.dropna(subset=["State", "ServiceType"]).groupby(
        ["State", "ServiceType"], observed=True
    )
    for (state, service_type), group in grouped:
        rated = group[group["is_rated"]]
        below = rated[rated["is_below_nqs"]]
        records.append(
            {
                "state": state,
                "service_type": service_type,
                "services": len(group),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": len(below),
                "below_nqs_pct_of_rated": _percent(len(below), len(rated)),
                "no_current_rating_services": len(group) - len(rated),
                "no_current_rating_pct": _percent(len(group) - len(rated), len(group)),
                "approved_places_in_below_nqs_services": below["NumberOfApprovedPlaces"].sum(min_count=1),
                "median_nearest_transport_km": group["nearest_transport_distance_km"].median(),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["below_nqs_services", "below_nqs_pct_of_rated"], ascending=False
    ).reset_index(drop=True)


def build_rating_composition(data, group_column, output_column):
    """Return 100% rating compositions for one categorical dimension."""
    records = []
    rated = data[data["is_rated"] & data[group_column].notna()]
    for group_name, group in rated.groupby(group_column, observed=True):
        counts = group["OverallRating"].value_counts()
        for rating in RATING_ORDER:
            count = int(counts.get(rating, 0))
            records.append(
                {
                    output_column: group_name,
                    "rating": rating,
                    "services": count,
                    "share_of_rated_pct": _percent(count, len(group)),
                }
            )
    return pd.DataFrame(records)


def build_quality_benchmark_by_state(data):
    """Compare state Meeting+ rates with the national benchmark."""
    measures = [("Overall", "OverallRating")] + [
        (QUALITY_AREA_LABELS[column], column) for column in QUALITY_AREA_COLUMNS
    ]
    records = []
    for measure, column in measures:
        national_rated = data[column].isin(RATING_ORDER)
        national_meeting = national_rated & ~data[column].isin(LOW_RATINGS)
        national_rate = _percent(int(national_meeting.sum()), int(national_rated.sum()))
        for state in STATE_ORDER:
            group = data[data["State"].eq(state)]
            rated = group[column].isin(RATING_ORDER)
            meeting = rated & ~group[column].isin(LOW_RATINGS)
            state_rate = _percent(int(meeting.sum()), int(rated.sum()))
            records.append(
                {
                    "state": state,
                    "measure": measure,
                    "rated_services": int(rated.sum()),
                    "meeting_or_above_services": int(meeting.sum()),
                    "meeting_or_above_pct": state_rate,
                    "national_meeting_or_above_pct": national_rate,
                    "gap_from_national_pp": state_rate - national_rate,
                }
            )
    return pd.DataFrame(records)


def build_transport_by_remoteness(data):
    """Summarise separate bus, train, and nearest transport distances."""
    modes = {
        "Bus": "DistanceToBusStation_km",
        "Train": "DistanceToTrainStation_km",
        "Nearest listed mode": "nearest_transport_distance_km",
    }
    records = []
    for area in REMOTENESS_ORDER:
        group = data[data["remoteness_area"].eq(area)]
        for mode, column in modes.items():
            values = group.loc[group[column].ge(0), column].dropna()
            records.append(
                {
                    "remoteness_area": area,
                    "transport_mode": mode,
                    "services": len(values),
                    "median_distance_km": values.median(),
                    "q1_distance_km": values.quantile(0.25),
                    "q3_distance_km": values.quantile(0.75),
                    "p90_distance_km": values.quantile(0.90),
                }
            )
    return pd.DataFrame(records)


def build_accessibility_quality_quintiles(data):
    """Relate global transport-distance quintiles to Meeting+ rates."""
    modes = {
        "Bus": "DistanceToBusStation_km",
        "Train": "DistanceToTrainStation_km",
    }
    scopes = [
        ("All services", pd.Series(True, index=data.index)),
        ("Centre-Based Care", data["ServiceType"].eq("Centre-Based Care")),
        ("Family Day Care", data["ServiceType"].eq("Family Day Care")),
    ] + [(area, data["remoteness_area"].eq(area)) for area in REMOTENESS_ORDER]
    records = []
    for mode, column in modes.items():
        valid = data[column].notna() & data[column].ge(0)
        quintile = pd.Series(pd.NA, index=data.index, dtype="Int64")
        quintile.loc[valid] = (
            pd.qcut(
                data.loc[valid, column].rank(method="first"),
                q=5,
                labels=False,
            )
            + 1
        ).astype("Int64")
        for scope_name, scope_mask in scopes:
            for quintile_number in range(1, 6):
                group = data[
                    scope_mask & quintile.eq(quintile_number).fillna(False)
                ]
                rated = group[group["is_rated"]]
                successes = int(rated["is_meeting_or_above"].sum())
                lower, upper = _wilson_interval(successes, len(rated))
                records.append(
                    {
                        "scope": scope_name,
                        "transport_mode": mode,
                        "distance_quintile": quintile_number,
                        "services": len(group),
                        "rated_services": len(rated),
                        "median_distance_km": group[column].median(),
                        "meeting_or_above_services": successes,
                        "meeting_or_above_pct": _percent(successes, len(rated)),
                        "ci95_lower_pct": lower,
                        "ci95_upper_pct": upper,
                    }
                )
    return pd.DataFrame(records)


def build_operations_by_remoteness(data):
    """Summarise like-for-like capacity and annual operating hours."""
    records = []
    for area in REMOTENESS_ORDER:
        for service_type in ["Centre-Based Care", "Family Day Care"]:
            group = data[
                data["remoteness_area"].eq(area)
                & data["ServiceType"].eq(service_type)
            ]
            capacity = group["NumberOfApprovedPlaces"].dropna()
            hours = group["annual_weekly_operating_hours"].dropna()
            records.append(
                {
                    "remoteness_area": area,
                    "service_type": service_type,
                    "services": len(group),
                    "capacity_services": len(capacity),
                    "median_approved_places": capacity.median(),
                    "q1_approved_places": capacity.quantile(0.25),
                    "q3_approved_places": capacity.quantile(0.75),
                    "annual_hours_services": len(hours),
                    "median_annual_weekly_hours": hours.median(),
                    "q1_annual_weekly_hours": hours.quantile(0.25),
                    "q3_annual_weekly_hours": hours.quantile(0.75),
                }
            )
    return pd.DataFrame(records)


def build_approval_year_by_service_type(data):
    valid = data.dropna(subset=["approval_year", "ServiceType"]).copy()
    valid["approval_year"] = valid["approval_year"].astype(int)
    return (
        valid.groupby(["approval_year", "ServiceType"], observed=True)
        .size()
        .rename("current_services")
        .reset_index()
        .rename(columns={"ServiceType": "service_type"})
    )


def build_compound_disadvantage(data):
    """Screen Centre-Based State x Remoteness groups on compatible denominators."""
    centre_data = data[data["ServiceType"].eq("Centre-Based Care")].copy()
    national_rated = centre_data[centre_data["is_rated"]]
    national_below = _percent(
        int(national_rated["is_below_nqs"].sum()), len(national_rated)
    )
    national_transport = centre_data["nearest_transport_distance_km"].median()
    national_capacity = centre_data["NumberOfApprovedPlaces"].median()

    records = []
    classified = centre_data.dropna(subset=["State", "remoteness_area"])
    for (state, area), group in classified.groupby(
        ["State", "remoteness_area"], observed=True
    ):
        rated = group[group["is_rated"]]
        annual_hours = group["annual_weekly_operating_hours"].dropna()
        below_count = int(rated["is_below_nqs"].sum())
        below_rate = _percent(below_count, len(rated))
        below_lower, below_upper = _wilson_interval(below_count, len(rated))
        transport = group["nearest_transport_distance_km"].median()
        capacity = group["NumberOfApprovedPlaces"].median()
        eligible = (
            len(rated) >= 20
            and len(group) >= 20
            and pd.notna(below_rate)
            and pd.notna(transport)
            and pd.notna(capacity)
        )
        screening_flag = bool(
            eligible
            and below_rate > national_below
            and transport > national_transport
            and capacity < national_capacity
        )
        records.append(
            {
                "state": state,
                "remoteness_area": area,
                "services": len(group),
                "rated_services": len(rated),
                "below_nqs_pct_of_rated": below_rate,
                "below_nqs_ci95_lower_pct": below_lower,
                "below_nqs_ci95_upper_pct": below_upper,
                "median_nearest_transport_km": transport,
                "median_approved_places": capacity,
                "annual_hours_services": len(annual_hours),
                "median_annual_weekly_hours": annual_hours.median(),
                "eligible_for_comparison": eligible,
                "quality_disadvantage": eligible and below_rate > national_below,
                "access_disadvantage": eligible and transport > national_transport,
                "capacity_disadvantage": eligible and capacity < national_capacity,
                "screening_flag": screening_flag,
                "national_below_nqs_pct": national_below,
                "national_median_transport_km": national_transport,
                "national_median_approved_places": national_capacity,
            }
        )
    return pd.DataFrame(records)


def _service_filter_masks(data):
    return [
        ("All services", pd.Series(True, index=data.index)),
        ("Centre-Based Care", data["ServiceType"].eq("Centre-Based Care")),
        ("Family Day Care", data["ServiceType"].eq("Family Day Care")),
    ]


def build_offering_summary(data):
    """Summarise overlapping detailed service offerings."""
    records = []
    centre_total = int(data["ServiceType"].eq("Centre-Based Care").sum())
    for column, label in DETAILED_SERVICE_OFFERINGS.items():
        group = data[data[column].eq("Yes")]
        rated = group[group["is_rated"]]
        below = int(rated["is_below_nqs"].sum())
        lower, upper = _wilson_interval(below, len(rated))
        capacity = group["NumberOfApprovedPlaces"].dropna()
        annual_hours = group["annual_weekly_operating_hours"].dropna()
        records.append(
            {
                "offering": label,
                "source_column": column,
                "services": len(group),
                "share_all_services_pct": _percent(len(group), len(data)),
                "share_centre_based_pct": _percent(len(group), centre_total),
                "rated_services": len(rated),
                "rating_coverage_pct": _percent(len(rated), len(group)),
                "below_nqs_services": below,
                "below_nqs_pct_of_rated": _percent(below, len(rated)),
                "below_ci95_lower_pct": lower,
                "below_ci95_upper_pct": upper,
                "capacity_services": len(capacity),
                "median_approved_places": capacity.median(),
                "q1_approved_places": capacity.quantile(0.25),
                "q3_approved_places": capacity.quantile(0.75),
                "annual_hours_services": len(annual_hours),
                "median_annual_weekly_hours": annual_hours.median(),
                "q1_annual_weekly_hours": annual_hours.quantile(0.25),
                "q3_annual_weekly_hours": annual_hours.quantile(0.75),
                "median_bus_distance_km": group["DistanceToBusStation_km"].median(),
                "median_train_distance_km": group["DistanceToTrainStation_km"].median(),
                "median_nearest_transport_km": group[
                    "nearest_transport_distance_km"
                ].median(),
            }
        )
    result = pd.DataFrame(records)
    result["offering_order"] = result["offering"].map(
        {name: position for position, name in enumerate(OFFERING_ORDER)}
    )
    return result.sort_values("offering_order").drop(columns="offering_order")


def build_offering_combinations(data):
    counts = data["detailed_offering_combination"].value_counts(dropna=False)
    result = counts.rename_axis("offering_combination").rename("services").reset_index()
    result["share_all_services_pct"] = result["services"].div(len(data)).mul(100)
    result["number_of_offerings"] = result["offering_combination"].where(
        ~result["offering_combination"].eq("No detailed offering recorded"), ""
    ).str.count(r"\+").add(1).where(
        ~result["offering_combination"].eq("No detailed offering recorded"), 0
    )
    return result


def build_offering_by_remoteness(data):
    records = []
    for column, label in DETAILED_SERVICE_OFFERINGS.items():
        national_count = int(data[column].eq("Yes").sum())
        national_share = _percent(national_count, len(data))
        for area in REMOTENESS_ORDER:
            group = data[data["remoteness_area"].eq(area)]
            count = int(group[column].eq("Yes").sum())
            share = _percent(count, len(group))
            records.append(
                {
                    "offering": label,
                    "remoteness_area": area,
                    "services": count,
                    "services_in_area": len(group),
                    "share_of_area_services_pct": share,
                    "national_share_pct": national_share,
                    "gap_from_national_pp": share - national_share,
                }
            )
    return pd.DataFrame(records)


def build_filtered_quality_benchmark(data):
    measures = [("Overall", "OverallRating")] + [
        (QUALITY_AREA_LABELS[column], column) for column in QUALITY_AREA_COLUMNS
    ]
    records = []
    for filter_name, filter_mask in _service_filter_masks(data):
        subset = data[filter_mask]
        for measure, column in measures:
            national_rated = subset[column].isin(RATING_ORDER)
            national_meeting = national_rated & ~subset[column].isin(LOW_RATINGS)
            national_rate = _percent(
                int(national_meeting.sum()), int(national_rated.sum())
            )
            for state in STATE_ORDER:
                group = subset[subset["State"].eq(state)]
                rated = group[column].isin(RATING_ORDER)
                meeting = rated & ~group[column].isin(LOW_RATINGS)
                state_rate = _percent(int(meeting.sum()), int(rated.sum()))
                records.append(
                    {
                        "service_filter": filter_name,
                        "services_in_filter": len(subset),
                        "state": state,
                        "measure": measure,
                        "rated_services": int(rated.sum()),
                        "meeting_or_above_pct": state_rate,
                        "filter_national_meeting_or_above_pct": national_rate,
                        "gap_from_filter_national_pp": state_rate - national_rate,
                    }
                )
    return pd.DataFrame(records)


def build_filtered_compound_disadvantage(data):
    filters = [("Centre-Based Care", data["ServiceType"].eq("Centre-Based Care"))]
    filters.extend(
        (label, data[column].eq("Yes"))
        for column, label in DETAILED_SERVICE_OFFERINGS.items()
    )
    records = []
    for filter_name, mask in filters:
        subset = data[mask & data["remoteness_area"].notna()]
        national_rated = subset[subset["is_rated"]]
        national_below = _percent(
            int(national_rated["is_below_nqs"].sum()), len(national_rated)
        )
        national_transport = subset["nearest_transport_distance_km"].median()
        national_capacity = subset["NumberOfApprovedPlaces"].median()
        for (state, area), group in subset.groupby(
            ["State", "remoteness_area"], observed=True
        ):
            rated = group[group["is_rated"]]
            below_rate = _percent(int(rated["is_below_nqs"].sum()), len(rated))
            transport = group["nearest_transport_distance_km"].median()
            capacity = group["NumberOfApprovedPlaces"].median()
            eligible = len(group) >= 20 and len(rated) >= 20
            compound = bool(
                eligible
                and pd.notna(below_rate)
                and pd.notna(transport)
                and pd.notna(capacity)
                and below_rate > national_below
                and transport > national_transport
                and capacity < national_capacity
            )
            records.append(
                {
                    "service_filter": filter_name,
                    "state": state,
                    "remoteness_area": area,
                    "services": len(group),
                    "rated_services": len(rated),
                    "below_nqs_pct_of_rated": below_rate,
                    "median_nearest_transport_km": transport,
                    "median_approved_places": capacity,
                    "median_annual_weekly_hours": group[
                        "annual_weekly_operating_hours"
                    ].median(),
                    "eligible_for_comparison": eligible,
                    "compound_disadvantage": compound,
                    "filter_national_below_nqs_pct": national_below,
                    "filter_national_median_transport_km": national_transport,
                    "filter_national_median_places": national_capacity,
                }
            )
    return pd.DataFrame(records)


def _write_findings(data, tables, output_path):
    state = tables["state_summary"]
    qa = tables["quality_area_summary"]
    remoteness = tables["remoteness_summary"]
    transport = tables["transport_by_remoteness"]
    relation = tables["accessibility_quality_quintiles"]
    operations = tables["operations_by_remoteness"]
    compound = tables["compound_disadvantage"]
    offering_mix = tables["offering_by_remoteness"]
    quality_models = tables["quality_model_coefficients"]
    operational_models = tables["operational_model_effects"]
    spatial = tables["spatial_join_diagnostics"].set_index("metric")["count"]
    coverage = tables["population_coverage_by_remoteness"]
    national_coverage = tables["population_coverage_national"].iloc[0]

    highest_state = state.loc[state["below_nqs_pct_of_rated"].idxmax()]
    weakest_qa = qa.loc[qa["below_nqs_pct_of_rated"].idxmax()]
    remote_high = remoteness.loc[remoteness["below_nqs_pct_of_rated"].idxmax()]
    remote_low = remoteness.loc[remoteness["below_nqs_pct_of_rated"].idxmin()]
    very_remote_access = transport[
        transport["remoteness_area"].eq("Very Remote Australia")
        & transport["transport_mode"].eq("Nearest listed mode")
    ].iloc[0]
    city_access = transport[
        transport["remoteness_area"].eq("Major Cities of Australia")
        & transport["transport_mode"].eq("Nearest listed mode")
    ].iloc[0]
    overall_relation = relation[relation["scope"].eq("All services")]
    q1 = overall_relation[overall_relation["distance_quintile"].eq(1)]
    q5 = overall_relation[overall_relation["distance_quintile"].eq(5)]
    centre_ops = operations[operations["service_type"].eq("Centre-Based Care")]
    compound_groups = compound[compound["enhanced_screening_flag"]]
    city_coverage = coverage[
        coverage["remoteness_area"].eq("Major Cities of Australia")
    ].iloc[0]
    very_remote_coverage = coverage[
        coverage["remoteness_area"].eq("Very Remote Australia")
    ].iloc[0]
    largest_mix_shift = offering_mix.loc[
        offering_mix["gap_from_national_pp"].abs().idxmax()
    ]
    capacity_low = centre_ops.loc[centre_ops["median_approved_places"].idxmin()]
    capacity_high = centre_ops.loc[centre_ops["median_approved_places"].idxmax()]

    relationship_lines = []
    for mode in ["Bus", "Train"]:
        start = q1[q1["transport_mode"].eq(mode)].iloc[0]
        end = q5[q5["transport_mode"].eq(mode)].iloc[0]
        adjusted = quality_models[
            quality_models["transport_mode"].eq(mode)
        ].iloc[0]
        relationship_lines.append(
            "- {}: the unadjusted Meeting+ rate is {:.1f}% in Q1 and {:.1f}% in Q5. After controlling for remoteness, state, and broad service type, the odds ratio per distance doubling is {:.3f} (95% CI {:.3f}-{:.3f}, n={:,}).".format(
                mode,
                start["meeting_or_above_pct"],
                end["meeting_or_above_pct"],
                adjusted["odds_ratio_per_distance_doubling"],
                adjusted["or_ci95_lower"],
                adjusted["or_ci95_upper"],
                int(adjusted["services"]),
            )
        )

    operational_lines = []
    for outcome in ["Centre-Based approved capacity", "Annual weekly opening hours"]:
        for mode in ["Bus", "Train"]:
            row = operational_models[
                operational_models["outcome"].eq(outcome)
                & operational_models["transport_mode"].eq(mode)
            ].iloc[0]
            operational_lines.append(
                "- {} x {} distance: {:.2f} {} per distance doubling (95% CI {:.2f} to {:.2f}, n={:,}), adjusted for geography{}.".format(
                    outcome,
                    mode.lower(),
                    row["adjusted_effect_per_distance_doubling"],
                    row["effect_unit"],
                    row["ci95_lower"],
                    row["ci95_upper"],
                    int(row["services"]),
                    " and broad service type" if outcome == "Annual weekly opening hours" else "",
                )
            )

    lines = [
        "# Findings organised by the six-stage analytical story",
        "",
        "These results are descriptive associations from a current service-register snapshot. They do not establish causality or estimate unmet demand.",
        "",
        "## 1. National landscape",
        "",
        "- The dataset contains {:,} services; {:,} have valid coordinates and {:,} were matched to an ABS 2021 Remoteness Area.".format(
            len(data), int(data["coordinate_valid"].sum()), int(spatial["Matched to ABS remoteness area"])
        ),
        "- The static and interactive grid maps aggregate observed service locations into equal-area cells; they do not measure services or places per child.",
        "- The largest service-mix departure is {} in {} ({:+.1f} percentage points from national prevalence, n={:,}); service composition is therefore treated as a potential confounder rather than the main outcome.".format(
            largest_mix_shift["offering"],
            largest_mix_shift["remoteness_area"],
            largest_mix_shift["gap_from_national_pp"],
            int(largest_mix_shift["services"]),
        ),
        "",
        "## 2. Quality",
        "",
        "- {} has the highest below-NQS share among rated services ({:.1f}%).".format(
            highest_state["state"], highest_state["below_nqs_pct_of_rated"]
        ),
        "- {} ({}) has the highest national below-NQS rate ({:.1f}%) across the seven quality areas.".format(
            weakest_qa["quality_area"], weakest_qa["quality_area_name"], weakest_qa["below_nqs_pct_of_rated"]
        ),
        "- Across remoteness groups, below-NQS rates range from {:.1f}% in {} to {:.1f}% in {}; these are unadjusted geographic comparisons.".format(
            remote_low["below_nqs_pct_of_rated"],
            remote_low["remoteness_area"],
            remote_high["below_nqs_pct_of_rated"],
            remote_high["remoteness_area"],
        ),
        "- The quality benchmark heatmap reports percentage-point gaps from the national Meeting+ rate, rather than ranking jurisdictions by raw counts.",
        "",
        "## 3. Accessibility and observed coverage",
        "",
        "- Median distance to the nearest listed bus or train service rises from {:.2f} km in Major Cities to {:.2f} km in Very Remote Australia.".format(
            city_access["median_distance_km"], very_remote_access["median_distance_km"]
        ),
        "- Bus and train are kept separate because absence of rail in remote areas is not equivalent to absence of all transport.",
        "- Population adjustment changes the coverage interpretation: Centre-Based approved places per 1,000 children aged 0-13 decline from {:.1f} in Major Cities to {:.1f} in Very Remote Australia, versus a national rate of {:.1f}.".format(
            city_coverage["approved_places_per_1000_children"],
            very_remote_coverage["approved_places_per_1000_children"],
            national_coverage["approved_places_per_1000_children"],
        ),
        "- Centre-Based service counts per 1,000 children vary much less than approved places per 1,000, indicating that facility size and service mix contribute to the observed capacity-intensity gap.",
        "",
        "## 4. Quality x accessibility",
        "",
    ] + relationship_lines + [
        "- Models describe adjusted associations, not causal effects. Odds ratios close to 1 indicate that much of the raw gradient is geographic or compositional.",
        "",
        "## 5. Operations",
        "",
        "- Centre-Based median capacity ranges from {:.1f} places in {} to {:.1f} places in {}; Family Day Care is excluded from capacity comparisons because approved places are unavailable.".format(
            capacity_low["median_approved_places"],
            capacity_low["remoteness_area"],
            capacity_high["median_approved_places"],
            capacity_high["remoteness_area"],
        ),
        "- Annual weekly operating hours are based only on the annual timetable ({:,} services); school-term and holiday schedules remain separate derived variables.".format(
            int(data["annual_weekly_operating_hours"].notna().sum())
        ),
    ] + operational_lines + [
        "- Current services by approval year describe the present register's approval cohorts, not historical sector growth.",
        "",
        "## 6. Synthesis",
        "",
        "- {:,} Centre-Based State x Remoteness groups meet the enhanced screening rule: above-national below-NQS rate and transport distance, below-national median capacity, and below-national approved places per 1,000 children, with at least 20 rated services.".format(
            len(compound_groups)
        ),
        "- Screening flags identify groups warranting further investigation; they are not proof of disadvantage and do not form a weighted priority score.",
        "",
        "## Current limitations",
        "",
        "- The denominator is the 2021 Census population aged 0-13, while the service register is a later current snapshot; the ratio is a planning proxy with temporal mismatch, not a contemporaneous coverage estimate.",
        "- Census counts are perturbed for confidentiality, and broad ages 0-13 do not align perfectly with the age eligibility of every Long Day Care, preschool, OSHC, or Family Day Care offering.",
        "- Approved places exclude Family Day Care and do not measure occupied places, vacancies, attendance, staffing capacity, or unmet demand.",
        "- Public transport distance does not measure travel time, frequency, affordability, or actual family access.",
        "- Remoteness describes the service location, not family residence or catchment.",
        "- Operating calendars are heterogeneous; annual, school-term, and holiday schedules must not be added together.",
        "- State x Remote and Very Remote groups can have small rated samples; rates and screening flags should be read with n and confidence intervals.",
        "- Ratings and operational measures are cross-sectional and cannot identify causes or changes over time.",
    ]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analysis_outputs(data, output_dir, child_population=None):
    """Build and export all summary tables used by the figures and report."""
    paths = output_paths(output_dir)
    tables = {
        "data_quality_summary": build_data_quality_summary(data),
        "eda_descriptive_statistics": build_eda_descriptive_statistics(data),
        "overall_rating_summary": build_overall_rating_summary(data),
        "state_summary": build_state_summary(data),
        "quality_area_summary": build_quality_area_summary(data),
        "remoteness_summary": build_remoteness_summary(data),
        "remoteness_rating_composition": build_remoteness_rating_composition(data),
        "quality_area_by_remoteness": build_quality_area_by_remoteness(data),
        "remoteness_service_type_summary": build_remoteness_service_type_summary(data),
        "spatial_join_diagnostics": build_spatial_join_diagnostics(data),
        "state_rating_composition": build_rating_composition(
            data[data["State"].isin(STATE_ORDER)], "State", "state"
        ),
        "service_type_rating_composition": build_rating_composition(
            data, "ServiceType", "service_type"
        ),
        "quality_benchmark_by_state": build_quality_benchmark_by_state(data),
        "transport_by_remoteness": build_transport_by_remoteness(data),
        "accessibility_quality_quintiles": build_accessibility_quality_quintiles(data),
        "operations_by_remoteness": build_operations_by_remoteness(data),
        "approval_year_by_service_type": build_approval_year_by_service_type(data),
        "offering_summary": build_offering_summary(data),
        "offering_combinations": build_offering_combinations(data),
        "offering_by_remoteness": build_offering_by_remoteness(data),
        "filtered_quality_benchmark": build_filtered_quality_benchmark(data),
    }
    screening = build_compound_disadvantage(data)
    if child_population is not None:
        population_tables = build_population_coverage_tables(data, child_population)
        tables.update(population_tables)
        screening = enrich_screening_with_population(
            screening,
            population_tables["population_coverage_by_state_remoteness"],
        )
        child_population.to_csv(
            paths["data"] / "abs_2021_child_population_0_13_by_ra.csv",
            index=False,
        )
    tables["compound_disadvantage"] = screening
    tables.update(build_model_outputs(data))

    for stale_table in paths["tables"].glob("*.csv"):
        stale_table.unlink()
    for name, table in tables.items():
        table.to_csv(paths["tables"] / "{}.csv".format(name), index=False, float_format="%.3f")

    classification_columns = [
        "ServiceApprovalNumber",
        "ServiceName",
        "ProviderLegalName",
        "ServiceType",
        "Suburb",
        "State",
        "Postcode",
        "longitude",
        "latitude",
        "coordinate_valid",
        "OverallRating",
        "rating_group",
        "NumberOfApprovedPlaces",
        "DistanceToBusStation_km",
        "DistanceToTrainStation_km",
        "nearest_transport_distance_km",
        "transport_distance_invalid",
        "transport_distance_review",
        "annual_weekly_operating_hours",
        "annual_hours_valid",
        "term_weekly_operating_hours_unadjusted",
        "term_weekly_operating_hours",
        "term_hours_overlap_adjusted",
        "holiday_weekly_operating_hours",
        "weekly_operating_hours",
        "operating_hours_basis",
        "detailed_offering_count",
        "detailed_offering_combination",
        "remoteness_code",
        "remoteness_area",
        "remoteness_broad_group",
        "boundary_state_name",
        "boundary_state_abbr",
        "remoteness_match_status",
        "state_boundary_consistent",
    ] + QUALITY_AREA_COLUMNS + list(DETAILED_SERVICE_OFFERINGS)
    data[classification_columns].to_csv(
        paths["data"] / "service_remoteness_classification.csv",
        index=False,
        float_format="%.6f",
    )

    _write_findings(data, tables, paths["report"] / "key_findings.md")
    return tables
