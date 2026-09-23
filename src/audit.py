"""Create reproducible source-data and analysis-sample audit outputs."""

from pathlib import Path

import pandas as pd

from config import (
    DETAILED_SERVICE_OFFERINGS,
    QUALITY_AREA_COLUMNS,
    RATING_ORDER,
    REMOTENESS_ORDER,
    STATE_ORDER,
    output_paths,
)


NUMERIC_COLUMNS = [
    "NumberOfApprovedPlaces",
    "DistanceToTrainStation_km",
    "DistanceToBusStation_km",
]


def _percent(count, total):
    return 100 * count / total if total else float("nan")


def _numeric_coercion_failures(raw, column):
    converted = pd.to_numeric(raw[column], errors="coerce")
    return int((raw[column].notna() & converted.isna()).sum())


def build_cleaning_summary(raw, data):
    """Document flags and decisions without implying global row deletion."""
    qa_unrecognised = sum(
        int((data[column].notna() & ~data[column].isin(RATING_ORDER)).sum())
        for column in QUALITY_AREA_COLUMNS
    )
    rows = [
        (
            "Dataset integrity",
            "Exact duplicate rows",
            int(raw.duplicated().sum()),
            "Flag for review before any removal; none detected",
            len(data),
        ),
        (
            "Dataset integrity",
            "Duplicate ServiceApprovalNumber",
            int(raw["ServiceApprovalNumber"].duplicated().sum()),
            "Investigate before deduplication; none detected",
            data["ServiceApprovalNumber"].nunique(),
        ),
        (
            "Quality",
            "Missing overall rating",
            int(data["overall_rating_missing"].sum()),
            "Retain service; exclude only from rated-outcome analyses",
            int(data["is_rated"].sum()),
        ),
        (
            "Quality",
            "Unrecognised overall rating",
            int(data["overall_rating_unrecognised"].sum()),
            "Retain service and flag; exclude from rated-outcome analyses",
            int(data["is_rated"].sum()),
        ),
        (
            "Quality areas",
            "Unrecognised QA rating cells",
            qa_unrecognised,
            "Exclude only the affected QA cell from that QA denominator",
            int(data[QUALITY_AREA_COLUMNS].isin(RATING_ORDER).sum().sum()),
        ),
        (
            "Capacity",
            "Missing approved places",
            int(data["NumberOfApprovedPlaces"].isna().sum()),
            "Expected for Family Day Care; capacity analyses use Centre-Based Care",
            int(data["NumberOfApprovedPlaces"].notna().sum()),
        ),
        (
            "Capacity",
            "Non-positive approved places",
            int(data["NumberOfApprovedPlaces"].le(0).sum()),
            "Treat as invalid for capacity analysis; none detected",
            int(data["NumberOfApprovedPlaces"].gt(0).sum()),
        ),
        (
            "Capacity",
            "Approved places over 300",
            int(data["NumberOfApprovedPlaces"].gt(300).sum()),
            "Retain after record-level review; values are plausible large centres/OSHC",
            int(data["NumberOfApprovedPlaces"].gt(0).sum()),
        ),
        (
            "Transport",
            "Negative or missing bus/train distance",
            int(data["transport_distance_invalid"].sum()),
            "Exclude only from the affected transport analysis",
            int((~data["transport_distance_invalid"]).sum()),
        ),
        (
            "Transport",
            "Bus or train distance over 500 km",
            int(data["transport_distance_review"].sum()),
            "Retain as legitimate geographic extremes after remoteness review",
            int((~data["transport_distance_invalid"]).sum()),
        ),
        (
            "Geometry",
            "Invalid or unparsed coordinate",
            int((~data["coordinate_valid"]).sum()),
            "Retain for non-spatial analyses; exclude from spatial analyses",
            int(data["coordinate_valid"].sum()),
        ),
        (
            "Geometry",
            "Not matched to ABS Remoteness Area",
            int(data["remoteness_area"].isna().sum()),
            "Retain for non-geographic analyses; exclude from remoteness comparisons",
            int(data["remoteness_area"].isin(REMOTENESS_ORDER).sum()),
        ),
        (
            "Geometry",
            "Source state differs from boundary state",
            int(data["state_boundary_consistent"].eq(False).fillna(False).sum()),
            "Retain and flag for record-level review; do not overwrite source state",
            int(data["state_boundary_consistent"].eq(True).sum()),
        ),
        (
            "Operating hours",
            "Term calendars with overlapping sessions",
            int(data["term_hours_overlap_adjusted"].sum()),
            "Use the union of daily intervals to avoid double-counting overlap",
            int(data["term_weekly_operating_hours"].notna().sum()),
        ),
        (
            "Dates",
            "Missing or unparsed approval date",
            int(data["approval_date"].isna().sum()),
            "Retain service; exclude only from approval-cohort analysis",
            int(data["approval_date"].notna().sum()),
        ),
    ]
    numeric_rows = [
        (
            "Numeric conversion",
            f"Non-numeric values in {column}",
            _numeric_coercion_failures(raw, column),
            "Coerce invalid non-missing values to missing; exclude only from affected analysis",
            int(data[column].notna().sum()),
        )
        for column in NUMERIC_COLUMNS
    ]
    rows[2:2] = numeric_rows
    result = pd.DataFrame(
        rows,
        columns=[
            "scope",
            "check",
            "records_flagged",
            "decision",
            "records_available_after_decision",
        ],
    )
    result.insert(3, "share_of_source_rows_pct", result["records_flagged"].map(lambda value: _percent(value, len(raw))))
    result.insert(4, "source_rows_before", len(raw))
    return result


def build_missingness_summary(data):
    handling = {
        "State": "Required for state-adjusted/geographic analyses only",
        "OverallRating": "Missing services remain in non-quality analyses",
        "NumberOfApprovedPlaces": "Capacity restricted to Centre-Based Care",
        "approval_date": "Required only for current-service cohort analysis",
        "rating_date": "Descriptive metadata; not used as an outcome",
        "longitude": "Required only for spatial analyses",
        "latitude": "Required only for spatial analyses",
        "remoteness_area": "Required only for remoteness comparisons/models",
        "DistanceToBusStation_km": "Required only for bus-distance analyses",
        "DistanceToTrainStation_km": "Required only for train-distance analyses",
        "annual_weekly_operating_hours": "Primary comparable hours measure",
        "term_weekly_operating_hours": "Kept separate from annual/holiday calendars",
        "holiday_weekly_operating_hours": "Kept separate from annual/term calendars",
    }
    records = []
    for column, decision in handling.items():
        missing = int(data[column].isna().sum())
        records.append(
            {
                "variable": column,
                "missing": missing,
                "missing_pct": _percent(missing, len(data)),
                "analysis_handling": decision,
            }
        )
    return pd.DataFrame(records)


def build_category_audit(data):
    expected = {
        "ServiceType": {"Centre-Based Care", "Family Day Care"},
        "OverallRating": set(RATING_ORDER),
        "State": set(STATE_ORDER),
    }
    expected.update({column: {"Yes", "No"} for column in DETAILED_SERVICE_OFFERINGS})
    records = []
    for column, allowed in expected.items():
        counts = data[column].fillna("<Missing>").value_counts(dropna=False)
        for value, count in counts.items():
            status = "Missing" if value == "<Missing>" else (
                "Recognised" if value in allowed else "Unrecognised"
            )
            records.append(
                {
                    "variable": column,
                    "value": value,
                    "count": int(count),
                    "status": status,
                }
            )
    for column in QUALITY_AREA_COLUMNS:
        counts = data[column].fillna("<Missing>").value_counts(dropna=False)
        for value, count in counts.items():
            status = "Missing" if value == "<Missing>" else (
                "Recognised" if value in RATING_ORDER else "Unrecognised"
            )
            records.append(
                {
                    "variable": column,
                    "value": value,
                    "count": int(count),
                    "status": status,
                }
            )
    return pd.DataFrame(records)


def build_transport_review(data):
    review = data[
        data["transport_distance_review"] | data["transport_distance_invalid"]
    ].copy()
    columns = [
        "ServiceApprovalNumber",
        "ServiceName",
        "State",
        "remoteness_area",
        "longitude",
        "latitude",
        "DistanceToBusStation_km",
        "DistanceToTrainStation_km",
        "transport_distance_invalid",
        "transport_distance_review",
    ]
    review = review[columns]
    review["decision"] = review["transport_distance_invalid"].map(
        {True: "Exclude from affected transport analysis", False: "Retain as reviewed geographic extreme"}
    )
    return review.sort_values(
        ["transport_distance_invalid", "DistanceToBusStation_km"],
        ascending=[False, False],
    )


def build_analysis_sample_flow(data):
    samples = [
        ("Source services", pd.Series(True, index=data.index)),
        ("Recognised overall rating", data["is_rated"]),
        ("Matched ABS remoteness", data["remoteness_area"].isin(REMOTENESS_ORDER)),
        (
            "Quality model complete cases",
            data["is_rated"]
            & data["State"].isin(STATE_ORDER)
            & data["remoteness_area"].isin(REMOTENESS_ORDER)
            & data["ServiceType"].notna()
            & ~data["transport_distance_invalid"],
        ),
        (
            "Centre-Based capacity analysis",
            data["ServiceType"].eq("Centre-Based Care")
            & data["NumberOfApprovedPlaces"].gt(0)
            & data["remoteness_area"].isin(REMOTENESS_ORDER),
        ),
        (
            "Annual-hours analysis",
            data["annual_hours_valid"]
            & data["remoteness_area"].isin(REMOTENESS_ORDER),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "analysis_sample": name,
                "services": int(mask.fillna(False).sum()),
                "share_of_source_pct": _percent(int(mask.fillna(False).sum()), len(data)),
            }
            for name, mask in samples
        ]
    )


def build_audit_outputs(source_path, data, output_dir):
    """Build and export all audit tables after geographic classification."""
    raw = pd.read_csv(Path(source_path), low_memory=False)
    raw.columns = raw.columns.str.strip()
    tables = {
        "cleaning_summary": build_cleaning_summary(raw, data),
        "missingness_summary": build_missingness_summary(data),
        "category_audit": build_category_audit(data),
        "transport_anomaly_review": build_transport_review(data),
        "analysis_sample_flow": build_analysis_sample_flow(data),
    }
    audit_dir = Path(output_paths(output_dir)["audit"])
    for stale in audit_dir.glob("*.csv"):
        stale.unlink()
    for name, table in tables.items():
        table.to_csv(audit_dir / f"{name}.csv", index=False, float_format="%.3f")
    return tables
