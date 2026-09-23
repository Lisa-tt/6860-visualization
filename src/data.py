"""Load, clean, validate, and derive fields from the source dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CAPACITY_BAND_ORDER,
    DETAILED_SERVICE_OFFERINGS,
    HIGH_RATINGS,
    LOW_RATINGS,
    QUALITY_AREA_COLUMNS,
    RATING_ORDER,
    TRANSPORT_BAND_ORDER,
)


TEXT_COLUMNS = [
    "ServiceApprovalNumber",
    "Provider Approval Number",
    "ServiceName",
    "ProviderLegalName",
    "ServiceType",
    "Suburb",
    "State",
    "Postcode",
    "OverallRating",
    "geometry",
]


ANNUAL_TIME_PAIRS = [
    (f"Annual {day} Start Time", f"Annual {day} End Time")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
]

TERM_TIME_PAIRS = [
    ("School Terms Only Session 1 Monday Start Time", "School Terms Only Session 1 Monday End Time"),
    ("School Terms Only Session 1 Tuesday Start Time", "School Terms Only Session 1 Tuesday End Time"),
    ("School Terms Only Session1 Wednesday Start Time", "School Terms Only Session1 Wednesday End Time"),
    ("School Terms Only Session1 Thursday Start Time", "School Terms Only Session1 Thursday End Time"),
    ("School Terms Only Session 1 Friday Start Time", "School Terms Only Session 1 Friday End Time"),
    ("School Terms Only Session 1 Saturday Start Time", "School Terms Only Session 1 Saturday End Time"),
    ("School Terms Only Session1 Sunday Start Time", "School Terms Only Session 1 Sunday End Time"),
    ("School Terms Only Session 2 Monday Start Time", "School Terms Only Session 2 Monday End Time"),
    ("School Terms Only Session 2 Tuesday Start Time", "School Terms Only Session 2 Tuesday End Time"),
    ("School Terms Only Session 2 Wednesday Start Time", "School Terms Only Session 2 Wednesday End Time"),
    ("School Terms Only Session 2 Thursday Start Time", "School Terms Only Session2 Thursday End Time"),
    ("School Terms Only Session 2 Friday Start Time", "School Terms Only Session 2 Friday End Time"),
    ("School Terms Only Session 2 Saturday Start Time", "School Terms Only Session 2 Saturday End Time"),
    ("School Terms Only Session 2 Sunday Start Time", "School Terms Only Session 2 Sunday End Time"),
]

HOLIDAY_TIME_PAIRS = [
    (f"Holiday Care {day} Start Time", f"Holiday Care {day} End Time")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
]

DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _clean_text(series):
    """Trim strings while preserving missing values."""
    return series.astype("string").str.strip().replace("", pd.NA)


def _parse_geometry(series):
    """Parse geometry strings in the form c(longitude, latitude)."""
    extracted = series.astype("string").str.extract(
        r"c\(\s*(?P<longitude>-?\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<latitude>-?\d+(?:\.\d+)?)\s*\)"
    )
    return extracted.apply(pd.to_numeric, errors="coerce")


def _rating_group(series):
    conditions = [
        series.isin(LOW_RATINGS),
        series.eq("Meeting NQS").fillna(False),
        series.isin(HIGH_RATINGS),
    ]
    conditions = [condition.to_numpy(dtype=bool) for condition in conditions]
    return pd.Series(
        np.select(
            conditions,
            ["Below NQS", "Meets NQS", "Above NQS"],
            default="No current rating",
        ),
        index=series.index,
        dtype="string",
    )


def _time_to_minutes(series):
    """Parse 24-hour clock strings such as 06:30 or 18:00:00."""
    parts = series.astype("string").str.strip().str.extract(
        r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::\d{2})?$"
    )
    hour = pd.to_numeric(parts["hour"], errors="coerce")
    minute = pd.to_numeric(parts["minute"], errors="coerce")
    valid = hour.between(0, 23) & minute.between(0, 59)
    return (hour * 60 + minute).where(valid)


def _weekly_hours(data, pairs):
    """Sum valid daily opening intervals for one operating calendar."""
    durations = []
    for start_column, end_column in pairs:
        start = _time_to_minutes(data[start_column])
        end = _time_to_minutes(data[end_column])
        duration = end - start
        duration = duration.where(duration.ge(0), duration + 24 * 60)
        duration = duration.where(duration.gt(0) & duration.le(24 * 60))
        durations.append(duration)
    return pd.concat(durations, axis=1).sum(axis=1, min_count=1).div(60)


def _union_interval_minutes(row):
    """Return the union of one day's intervals, avoiding double-counted overlaps."""
    values = row.to_numpy(dtype=float)
    segments = []
    for position in range(0, len(values), 2):
        start, end = values[position : position + 2]
        if np.isnan(start) or np.isnan(end) or start == end:
            continue
        if end > start:
            segments.append((start, end))
        else:
            segments.extend([(start, 24 * 60), (0, end)])
    if not segments:
        return np.nan
    merged = []
    for start, end in sorted(segments):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _weekly_hours_with_overlap_control(data, pairs):
    """Sum daily interval unions for calendars that can contain two sessions."""
    daily_durations = []
    for day in DAYS_OF_WEEK:
        day_pairs = [pair for pair in pairs if day in pair[0]]
        parsed = {}
        for pair_number, (start_column, end_column) in enumerate(day_pairs):
            parsed[f"start_{pair_number}"] = _time_to_minutes(data[start_column])
            parsed[f"end_{pair_number}"] = _time_to_minutes(data[end_column])
        daily = pd.DataFrame(parsed, index=data.index).apply(
            _union_interval_minutes, axis=1
        )
        daily_durations.append(daily)
    return pd.concat(daily_durations, axis=1).sum(axis=1, min_count=1).div(60)


def load_and_prepare_data(path):
    """Load the main CSV and return a cleaned analysis dataframe.

    The source columns remain available. Derived fields use snake_case names so
    that cleaning decisions are visible and do not overwrite the raw measures.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Source data not found: {}".format(path))

    data = pd.read_csv(path, dtype={"Postcode": "string"}, low_memory=False)
    data.columns = data.columns.str.strip()

    for column in TEXT_COLUMNS + QUALITY_AREA_COLUMNS + list(DETAILED_SERVICE_OFFERINGS):
        if column in data.columns:
            data[column] = _clean_text(data[column])

    data["State"] = data["State"].str.upper()
    data["Postcode"] = data["Postcode"].str.replace(r"\.0$", "", regex=True).str.zfill(4)

    numeric_columns = [
        "NumberOfApprovedPlaces",
        "DistanceToTrainStation_km",
        "DistanceToBusStation_km",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["approval_date"] = pd.to_datetime(
        data["ServiceApprovalGrantedDate"], dayfirst=True, errors="coerce"
    )
    data["rating_date"] = pd.to_datetime(
        data["RatingsIssued"], dayfirst=True, errors="coerce"
    )
    data["approval_year"] = data["approval_date"].dt.year.astype("Int64")
    data["rating_year"] = data["rating_date"].dt.year.astype("Int64")

    coordinates = _parse_geometry(data["geometry"])
    data["longitude"] = coordinates["longitude"]
    data["latitude"] = coordinates["latitude"]
    # Broad bounds retain Australian external territories while excluding
    # coordinates in the northern hemisphere or far outside the region.
    data["coordinate_valid"] = (
        data["longitude"].between(95, 170)
        & data["latitude"].between(-55, -9)
    )

    data["is_rated"] = data["OverallRating"].isin(RATING_ORDER)
    data["overall_rating_missing"] = data["OverallRating"].isna()
    data["overall_rating_unrecognised"] = (
        data["OverallRating"].notna() & ~data["OverallRating"].isin(RATING_ORDER)
    )
    data["is_below_nqs"] = data["OverallRating"].isin(LOW_RATINGS)
    data["is_above_nqs"] = data["OverallRating"].isin(HIGH_RATINGS)
    data["is_meeting_or_above"] = data["OverallRating"].isin(
        {"Meeting NQS"}.union(HIGH_RATINGS)
    )
    data["rating_group"] = _rating_group(data["OverallRating"])

    valid_distances = data[
        ["DistanceToTrainStation_km", "DistanceToBusStation_km"]
    ].where(lambda values: values.ge(0))
    data["nearest_transport_distance_km"] = valid_distances.min(axis=1)
    data["transport_distance_flag"] = (
        valid_distances.gt(500).any(axis=1)
        | data["nearest_transport_distance_km"].isna()
    )
    data["transport_distance_invalid"] = (
        data[["DistanceToTrainStation_km", "DistanceToBusStation_km"]]
        .lt(0)
        .any(axis=1)
        | data[["DistanceToTrainStation_km", "DistanceToBusStation_km"]]
        .isna()
        .any(axis=1)
    )
    data["transport_distance_review"] = valid_distances.gt(500).any(axis=1)
    data["transport_band"] = pd.cut(
        data["nearest_transport_distance_km"],
        bins=[0, 1, 2, 5, 10, np.inf],
        labels=TRANSPORT_BAND_ORDER,
        include_lowest=True,
        right=True,
    )

    centre_based_capacity = data["NumberOfApprovedPlaces"].where(
        data["ServiceType"].eq("Centre-Based Care")
    )
    data["capacity_band"] = pd.cut(
        centre_based_capacity,
        bins=[0, 40, 80, 120, np.inf],
        labels=CAPACITY_BAND_ORDER,
        include_lowest=True,
        right=True,
    )

    data["annual_weekly_operating_hours"] = _weekly_hours(data, ANNUAL_TIME_PAIRS)
    data["term_weekly_operating_hours_unadjusted"] = _weekly_hours(
        data, TERM_TIME_PAIRS
    )
    data["term_weekly_operating_hours"] = _weekly_hours_with_overlap_control(
        data, TERM_TIME_PAIRS
    )
    data["term_hours_overlap_adjusted"] = (
        data["term_weekly_operating_hours_unadjusted"].sub(
            data["term_weekly_operating_hours"]
        ).abs().gt(1e-6)
    )
    data["holiday_weekly_operating_hours"] = _weekly_hours(data, HOLIDAY_TIME_PAIRS)
    data["annual_hours_valid"] = data["annual_weekly_operating_hours"].between(
        0, 168, inclusive="both"
    )
    data["weekly_operating_hours"] = data[
        "annual_weekly_operating_hours"
    ].combine_first(data["term_weekly_operating_hours"]).combine_first(
        data["holiday_weekly_operating_hours"]
    )
    data["operating_hours_basis"] = np.select(
        [
            data["annual_weekly_operating_hours"].notna(),
            data["term_weekly_operating_hours"].notna(),
            data["holiday_weekly_operating_hours"].notna(),
        ],
        ["Annual", "School term", "Holiday care"],
        default="Missing",
    )

    data["approval_cohort"] = pd.cut(
        data["approval_year"].astype("float"),
        bins=[0, 2009, 2014, 2019, np.inf],
        labels=["Before 2010", "2010-2014", "2015-2019", "2020+"],
    )

    offering_flags = data[list(DETAILED_SERVICE_OFFERINGS)].eq("Yes")
    data["detailed_offering_count"] = offering_flags.sum(axis=1)
    labels = list(DETAILED_SERVICE_OFFERINGS.values())
    data["detailed_offering_combination"] = offering_flags.apply(
        lambda row: " + ".join(
            label for label, included in zip(labels, row.to_numpy()) if included
        )
        or "No detailed offering recorded",
        axis=1,
    ).astype("string")

    return data
