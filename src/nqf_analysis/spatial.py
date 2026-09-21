"""Spatially classify services using ABS 2021 Remoteness Areas."""

from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd

from .config import (
    BROAD_REMOTENESS_MAP,
    REMOTENESS_ORDER,
    STATE_NAME_TO_ABBR,
)


def _read_shapefile(path):
    """Read a shapefile with a fallback for GeoPandas/Fiona version mismatch."""
    try:
        return gpd.read_file(path)
    except AttributeError as error:
        # GeoPandas 0.13 calls fiona.path, which Fiona 1.10 removed.
        if "fiona" not in str(error).lower() and "path" not in str(error).lower():
            raise
        with fiona.open(path) as source:
            features = list(source)
            crs = source.crs_wkt or source.crs
        return gpd.GeoDataFrame.from_features(features, crs=crs)


def load_remoteness_boundaries(path):
    """Load and validate the official ABS 2021 GDA2020 remoteness boundaries."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "ABS remoteness shapefile not found: {}. Run "
            "scripts/download_abs_boundaries.py first.".format(path)
        )

    boundaries = _read_shapefile(path)
    required = {"RA_CODE21", "RA_NAME21", "STE_NAME21", "geometry"}
    missing = required.difference(boundaries.columns)
    if missing:
        raise ValueError("ABS boundary file is missing columns: {}".format(sorted(missing)))
    if boundaries.crs is None:
        raise ValueError("ABS remoteness boundary file has no coordinate reference system")

    boundaries = boundaries[
        ["RA_CODE21", "RA_NAME21", "STE_NAME21", "AREASQKM21", "geometry"]
    ].copy()
    boundaries = boundaries[boundaries["RA_NAME21"].isin(REMOTENESS_ORDER)].copy()
    boundaries["remoteness_area"] = pd.Categorical(
        boundaries["RA_NAME21"], categories=REMOTENESS_ORDER, ordered=True
    )
    boundaries["boundary_state_abbr"] = boundaries["STE_NAME21"].map(
        STATE_NAME_TO_ABBR
    )
    return boundaries


def classify_services_by_remoteness(data, boundaries):
    """Attach ABS remoteness fields using a point-in-polygon spatial join."""
    result = data.copy()
    result["remoteness_code"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["remoteness_area"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["remoteness_broad_group"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["boundary_state_name"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["boundary_state_abbr"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["remoteness_match_status"] = "Invalid coordinate"

    valid = result["coordinate_valid"]
    points = gpd.GeoDataFrame(
        result.loc[valid, ["State"]].copy(),
        geometry=gpd.points_from_xy(
            result.loc[valid, "longitude"], result.loc[valid, "latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs(boundaries.crs)

    join_columns = ["RA_CODE21", "RA_NAME21", "STE_NAME21", "geometry"]
    joined = gpd.sjoin(
        points,
        boundaries[join_columns],
        how="left",
        predicate="within",
    )
    if not joined.index.is_unique:
        duplicates = joined.index[joined.index.duplicated()].unique().tolist()
        raise ValueError(
            "Spatial join returned multiple remoteness areas for service rows: {}".format(
                duplicates[:10]
            )
        )

    result.loc[joined.index, "remoteness_code"] = joined["RA_CODE21"].astype("string")
    result.loc[joined.index, "remoteness_area"] = joined["RA_NAME21"].astype("string")
    result.loc[joined.index, "boundary_state_name"] = joined["STE_NAME21"].astype("string")
    result.loc[joined.index, "boundary_state_abbr"] = (
        joined["STE_NAME21"].map(STATE_NAME_TO_ABBR).astype("string")
    )
    result.loc[valid, "remoteness_match_status"] = "Valid coordinate outside RA boundary"
    matched_index = joined.index[joined["RA_NAME21"].notna()]
    result.loc[matched_index, "remoteness_match_status"] = "Matched by spatial join"
    result["remoteness_broad_group"] = result["remoteness_area"].map(
        BROAD_REMOTENESS_MAP
    ).astype("string")

    comparable_state = result["State"].notna() & result["boundary_state_abbr"].notna()
    result["state_boundary_consistent"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result.loc[comparable_state, "state_boundary_consistent"] = result.loc[
        comparable_state, "State"
    ].eq(result.loc[comparable_state, "boundary_state_abbr"])
    return result


def prepare_projected_services(data):
    """Project all valid service points once for repeated equal-area summaries."""
    selected = data[data["coordinate_valid"]].copy()
    return gpd.GeoDataFrame(
        selected,
        geometry=gpd.points_from_xy(selected["longitude"], selected["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3577")


def aggregate_projected_service_grid(projected, mask=None, cell_size_metres=75000):
    """Aggregate a prepared equal-area point layer for one selection."""
    if mask is None:
        mask = pd.Series(True, index=projected.index)
    projected = projected.loc[
        mask.reindex(projected.index, fill_value=False)
    ].copy()
    projected["projected_x"] = projected.geometry.x
    projected["projected_y"] = projected.geometry.y
    projected["grid_x"] = np.floor(
        projected["projected_x"] / cell_size_metres
    ).astype(int)
    projected["grid_y"] = np.floor(
        projected["projected_y"] / cell_size_metres
    ).astype(int)
    grouped = (
        projected.groupby(["grid_x", "grid_y"], observed=True)
        .agg(
            projected_x=("projected_x", "mean"),
            projected_y=("projected_y", "mean"),
            longitude=("longitude", "mean"),
            latitude=("latitude", "mean"),
            services=("ServiceApprovalNumber", "size"),
            rated_services=("is_rated", "sum"),
            below_nqs_services=("is_below_nqs", "sum"),
            median_transport_km=("nearest_transport_distance_km", "median"),
            median_capacity=("NumberOfApprovedPlaces", "median"),
            median_annual_hours=("annual_weekly_operating_hours", "median"),
        )
        .reset_index()
    )
    grouped["rating_coverage_pct"] = (
        grouped["rated_services"].div(grouped["services"]).mul(100)
    )
    grouped["below_nqs_pct"] = (
        grouped["below_nqs_services"]
        .div(grouped["rated_services"].replace(0, np.nan))
        .mul(100)
    )
    return grouped


def build_equal_area_service_grid(data, mask=None, cell_size_metres=75000):
    """Project and aggregate services to an Australian equal-area grid."""
    projected = prepare_projected_services(data)
    return aggregate_projected_service_grid(
        projected, mask=mask, cell_size_metres=cell_size_metres
    )
