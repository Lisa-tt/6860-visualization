"""Load ABS child population and build population-adjusted coverage metrics."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import REMOTENESS_ORDER, STATE_ORDER


ABS_G04A_MEMBER = (
    "2021 Census GCP Remoteness Areas for AUS/"
    "2021Census_G04A_AUST_RA.csv"
)
CHILD_AGE_COLUMNS = [f"Age_yr_{age}_P" for age in range(14)]


def load_child_population(path, area_column, population_column):
    """Load and validate a generic externally supplied population table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Child-population file not found: {path}")
    population = pd.read_csv(path)
    required = {area_column, population_column}
    missing = required.difference(population.columns)
    if missing:
        raise ValueError(f"Population file is missing columns: {sorted(missing)}")
    result = population[[area_column, population_column]].copy()
    result[population_column] = pd.to_numeric(
        result[population_column], errors="coerce"
    )
    if result[area_column].duplicated().any():
        raise ValueError("Population area keys must be unique before merging")
    if result[population_column].isna().any() or result[population_column].le(0).any():
        raise ValueError("Child-population denominators must be positive numeric values")
    return result


def load_abs_child_population(path, remoteness_boundaries):
    """Read exact ages 0-13 from the ABS 2021 GCP Remoteness Area DataPack."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ABS child-population DataPack not found: {path}. Run "
            "scripts/download_abs_child_population.py first."
        )

    with ZipFile(path) as archive:
        if ABS_G04A_MEMBER not in archive.namelist():
            raise ValueError(
                f"ABS DataPack does not contain expected member: {ABS_G04A_MEMBER}"
            )
        with archive.open(ABS_G04A_MEMBER) as source:
            raw = pd.read_csv(source, dtype={"RA_CODE_2021": "string"})

    required = {"RA_CODE_2021", *CHILD_AGE_COLUMNS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"ABS G04A table is missing required columns: {sorted(missing)}"
        )

    ages = raw[CHILD_AGE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if ages.isna().any().any() or ages.lt(0).any().any():
        raise ValueError("ABS age counts must be complete non-negative numeric values")

    population = pd.DataFrame(
        {
            "remoteness_code": raw["RA_CODE_2021"].str.replace(
                r"^RA", "", regex=True
            ),
            "child_population_0_13": ages.sum(axis=1),
        }
    )
    if population["remoteness_code"].duplicated().any():
        raise ValueError("ABS G04A Remoteness Area codes must be unique")

    geography = remoteness_boundaries[
        ["RA_CODE21", "remoteness_area", "boundary_state_abbr"]
    ].copy()
    geography = geography.rename(columns={"RA_CODE21": "remoteness_code"})
    geography["remoteness_code"] = geography["remoteness_code"].astype("string")
    geography["remoteness_area"] = geography["remoteness_area"].astype("string")
    geography = geography[
        geography["boundary_state_abbr"].isin(STATE_ORDER)
        & geography["remoteness_area"].isin(REMOTENESS_ORDER)
    ].drop_duplicates("remoteness_code")

    result = geography.merge(
        population,
        on="remoteness_code",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        missing_codes = result.loc[
            ~result["_merge"].eq("both"), "remoteness_code"
        ].tolist()
        raise ValueError(
            f"ABS population is missing regular Remoteness Area codes: {missing_codes}"
        )
    if result["child_population_0_13"].le(0).any():
        raise ValueError("Regular State x Remoteness populations must be positive")

    result = result.drop(columns="_merge").rename(
        columns={"boundary_state_abbr": "state"}
    )
    result["census_year"] = 2021
    result["population_definition"] = "2021 Census usual residents aged 0-13"
    state_order = {state: position for position, state in enumerate(STATE_ORDER)}
    remote_order = {
        area: position for position, area in enumerate(REMOTENESS_ORDER)
    }
    result["_state_order"] = result["state"].map(state_order)
    result["_remote_order"] = result["remoteness_area"].map(remote_order)
    return (
        result.sort_values(["_state_order", "_remote_order"])
        .drop(columns=["_state_order", "_remote_order"])
        .reset_index(drop=True)
    )


def add_population_adjusted_coverage(
    service_summary,
    population,
    area_column,
    approved_places_column,
    population_column,
):
    """Add approved places per 1,000 children for compatible geographic units."""
    required_service = {area_column, approved_places_column}
    required_population = {area_column, population_column}
    if missing := required_service.difference(service_summary.columns):
        raise ValueError(f"Service summary is missing columns: {sorted(missing)}")
    if missing := required_population.difference(population.columns):
        raise ValueError(f"Population table is missing columns: {sorted(missing)}")

    merged = service_summary.merge(
        population[[area_column, population_column]],
        on=area_column,
        how="left",
        validate="one_to_one",
        indicator="population_match_status",
    )
    merged["approved_places_per_1000_children"] = (
        merged[approved_places_column]
        .div(merged[population_column])
        .mul(1000)
    )
    return merged


def _coverage_rates(frame):
    result = frame.copy()
    denominator = result["child_population_0_13"].replace(0, np.nan)
    result["all_services_per_1000_children"] = (
        result["all_services"].div(denominator).mul(1000)
    )
    result["centre_based_services_per_1000_children"] = (
        result["centre_based_services"].div(denominator).mul(1000)
    )
    result["approved_places_per_1000_children"] = (
        result["centre_based_approved_places"].div(denominator).mul(1000)
    )
    result["approved_places_per_centre_service"] = result[
        "centre_based_approved_places"
    ].div(result["centre_based_services"].replace(0, np.nan))
    result["capacity_completeness_pct"] = (
        result["centre_based_capacity_records"]
        .div(result["centre_based_services"].replace(0, np.nan))
        .mul(100)
    )
    return result


def _aggregate_coverage(detail, group_columns):
    totals = [
        "child_population_0_13",
        "all_services",
        "centre_based_services",
        "centre_based_capacity_records",
        "centre_based_approved_places",
    ]
    if group_columns:
        result = (
            detail.groupby(group_columns, observed=True)[totals]
            .sum()
            .reset_index()
        )
    else:
        result = pd.DataFrame([detail[totals].sum()])
        result.insert(0, "geography", "Australia")
    return _coverage_rates(result)


def build_population_coverage_tables(data, population):
    """Build State x RA, remoteness, state, and national coverage tables."""
    service_geo = data[
        data["boundary_state_abbr"].isin(STATE_ORDER)
        & data["remoteness_area"].isin(REMOTENESS_ORDER)
        & data["remoteness_code"].notna()
    ].copy()
    service_geo["remoteness_code"] = service_geo["remoteness_code"].astype(
        "string"
    )

    all_services = (
        service_geo.groupby("remoteness_code", observed=True)
        .size()
        .rename("all_services")
    )
    centre = service_geo[service_geo["ServiceType"].eq("Centre-Based Care")]
    centre_summary = centre.groupby("remoteness_code", observed=True).agg(
        centre_based_services=("ServiceApprovalNumber", "size"),
        centre_based_capacity_records=("NumberOfApprovedPlaces", "count"),
        centre_based_approved_places=(
            "NumberOfApprovedPlaces",
            lambda values: values.sum(min_count=1),
        ),
    )

    detail = population.merge(
        all_services,
        left_on="remoteness_code",
        right_index=True,
        how="left",
        validate="one_to_one",
    ).merge(
        centre_summary,
        left_on="remoteness_code",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "all_services",
        "centre_based_services",
        "centre_based_capacity_records",
        "centre_based_approved_places",
    ]
    detail[count_columns] = detail[count_columns].fillna(0)
    detail = _coverage_rates(detail)

    national = _aggregate_coverage(detail, [])
    national_rate = float(national.loc[0, "approved_places_per_1000_children"])
    for frame in [detail, national]:
        frame["national_approved_places_per_1000_children"] = national_rate
        frame["coverage_gap_from_national"] = (
            frame["approved_places_per_1000_children"] - national_rate
        )

    by_remoteness = _aggregate_coverage(detail, ["remoteness_area"])
    by_state = _aggregate_coverage(detail, ["state"])
    for frame in [by_remoteness, by_state]:
        frame["national_approved_places_per_1000_children"] = national_rate
        frame["coverage_gap_from_national"] = (
            frame["approved_places_per_1000_children"] - national_rate
        )

    detail["coverage_disadvantage"] = detail[
        "approved_places_per_1000_children"
    ].lt(national_rate)
    detail["small_population_flag"] = detail["child_population_0_13"].lt(5000)

    remote_order = {
        area: position for position, area in enumerate(REMOTENESS_ORDER)
    }
    state_order = {state: position for position, state in enumerate(STATE_ORDER)}
    by_remoteness["_order"] = by_remoteness["remoteness_area"].map(remote_order)
    by_state["_order"] = by_state["state"].map(state_order)
    by_remoteness = by_remoteness.sort_values("_order").drop(columns="_order")
    by_state = by_state.sort_values("_order").drop(columns="_order")

    return {
        "child_population_reference": population.copy(),
        "population_coverage_by_state_remoteness": detail.reset_index(drop=True),
        "population_coverage_by_remoteness": by_remoteness.reset_index(drop=True),
        "population_coverage_by_state": by_state.reset_index(drop=True),
        "population_coverage_national": national.reset_index(drop=True),
    }


def enrich_screening_with_population(screening, coverage):
    """Add a population-adjusted coverage rule without replacing legacy rules."""
    columns = [
        "state",
        "remoteness_area",
        "child_population_0_13",
        "centre_based_approved_places",
        "approved_places_per_1000_children",
        "national_approved_places_per_1000_children",
        "coverage_gap_from_national",
    ]
    result = screening.merge(
        coverage[columns],
        on=["state", "remoteness_area"],
        how="left",
        validate="one_to_one",
    )
    result["population_coverage_available"] = result[
        "approved_places_per_1000_children"
    ].notna()
    result["coverage_disadvantage"] = (
        result["eligible_for_comparison"]
        & result["population_coverage_available"]
        & result["coverage_gap_from_national"].lt(0)
    )
    result["enhanced_screening_flag"] = (
        result["screening_flag"] & result["coverage_disadvantage"]
    )
    return result
