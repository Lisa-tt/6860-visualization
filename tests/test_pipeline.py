"""Focused tests for the highest-risk cleaning and summary rules."""

from pathlib import Path
import sys
import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import _wilson_interval, build_compound_disadvantage
from data import (
    TERM_TIME_PAIRS,
    _parse_geometry,
    _rating_group,
    _time_to_minutes,
    _weekly_hours_with_overlap_control,
)
from spatial import classify_services_by_remoteness
from population import add_population_adjusted_coverage
from population import (
    build_population_coverage_tables,
    enrich_screening_with_population,
)


class DataPreparationTests(unittest.TestCase):
    def test_geometry_parser(self):
        values = pd.Series(["c(151.2093, -33.8688)", "invalid", None])
        parsed = _parse_geometry(values)
        self.assertAlmostEqual(parsed.loc[0, "longitude"], 151.2093)
        self.assertAlmostEqual(parsed.loc[0, "latitude"], -33.8688)
        self.assertTrue(pd.isna(parsed.loc[1, "longitude"]))
        self.assertTrue(pd.isna(parsed.loc[2, "latitude"]))

    def test_rating_groups_keep_missing_separate(self):
        values = pd.Series(
            ["Working Towards NQS", "Meeting NQS", "Excellent", None], dtype="string"
        )
        result = _rating_group(values).tolist()
        self.assertEqual(
            result,
            ["Below NQS", "Meets NQS", "Above NQS", "No current rating"],
        )

    def test_time_parser_accepts_supported_clock_formats(self):
        values = pd.Series(["06:30", "18:15:00", "invalid", None])
        result = _time_to_minutes(values)
        self.assertEqual(result.iloc[0], 390)
        self.assertEqual(result.iloc[1], 1095)
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = _wilson_interval(10, 100)
        self.assertLess(lower, 10.0)
        self.assertGreater(upper, 10.0)

    def test_overlapping_term_sessions_are_not_double_counted(self):
        columns = sorted({column for pair in TERM_TIME_PAIRS for column in pair})
        data = pd.DataFrame({column: [pd.NA] for column in columns})
        data.loc[0, "School Terms Only Session 1 Monday Start Time"] = "06:00"
        data.loc[0, "School Terms Only Session 1 Monday End Time"] = "18:00"
        data.loc[0, "School Terms Only Session 2 Monday Start Time"] = "06:00"
        data.loc[0, "School Terms Only Session 2 Monday End Time"] = "18:00"
        result = _weekly_hours_with_overlap_control(data, TERM_TIME_PAIRS)
        self.assertEqual(result.iloc[0], 12.0)

    def test_synthesis_uses_centre_based_denominator_consistently(self):
        centre = pd.DataFrame(
            {
                "ServiceType": ["Centre-Based Care"] * 25,
                "is_rated": [True] * 25,
                "is_below_nqs": [False] * 25,
                "nearest_transport_distance_km": [2.0] * 25,
                "NumberOfApprovedPlaces": [60.0] * 25,
                "State": ["NSW"] * 25,
                "remoteness_area": ["Major Cities of Australia"] * 25,
                "annual_weekly_operating_hours": [50.0] * 25,
            }
        )
        family = centre.copy()
        family["ServiceType"] = "Family Day Care"
        family["is_below_nqs"] = True
        family["NumberOfApprovedPlaces"] = pd.NA
        result = build_compound_disadvantage(
            pd.concat([centre, family], ignore_index=True)
        )
        self.assertEqual(result.loc[0, "services"], 25)
        self.assertEqual(result.loc[0, "rated_services"], 25)
        self.assertEqual(result.loc[0, "below_nqs_pct_of_rated"], 0.0)

    def test_population_adjusted_coverage_uses_supplied_denominator(self):
        services = pd.DataFrame({"area": ["A"], "approved_places": [250]})
        population = pd.DataFrame({"area": ["A"], "children": [1000]})
        result = add_population_adjusted_coverage(
            services,
            population,
            area_column="area",
            approved_places_column="approved_places",
            population_column="children",
        )
        self.assertEqual(result.loc[0, "approved_places_per_1000_children"], 250.0)

    def test_population_coverage_uses_compatible_ra_numerator(self):
        services = pd.DataFrame(
            {
                "ServiceApprovalNumber": ["A", "B", "C"],
                "ServiceType": [
                    "Centre-Based Care",
                    "Centre-Based Care",
                    "Family Day Care",
                ],
                "NumberOfApprovedPlaces": [100.0, 150.0, pd.NA],
                "boundary_state_abbr": ["NSW", "NSW", "NSW"],
                "remoteness_area": ["Major Cities of Australia"] * 3,
                "remoteness_code": pd.Series(["10", "10", "10"], dtype="string"),
            }
        )
        population = pd.DataFrame(
            {
                "remoteness_code": pd.Series(["10"], dtype="string"),
                "remoteness_area": ["Major Cities of Australia"],
                "state": ["NSW"],
                "child_population_0_13": [1000],
                "census_year": [2021],
                "population_definition": ["Test population"],
            }
        )
        result = build_population_coverage_tables(services, population)[
            "population_coverage_by_state_remoteness"
        ]
        self.assertEqual(result.loc[0, "all_services"], 3)
        self.assertEqual(result.loc[0, "centre_based_services"], 2)
        self.assertEqual(result.loc[0, "centre_based_approved_places"], 250.0)
        self.assertEqual(result.loc[0, "approved_places_per_1000_children"], 250.0)

    def test_population_rule_enriches_without_replacing_original_screen(self):
        screening = pd.DataFrame(
            {
                "state": ["NSW"],
                "remoteness_area": ["Outer Regional Australia"],
                "eligible_for_comparison": [True],
                "screening_flag": [True],
            }
        )
        coverage = pd.DataFrame(
            {
                "state": ["NSW"],
                "remoteness_area": ["Outer Regional Australia"],
                "child_population_0_13": [1000],
                "centre_based_approved_places": [200],
                "approved_places_per_1000_children": [200.0],
                "national_approved_places_per_1000_children": [275.0],
                "coverage_gap_from_national": [-75.0],
            }
        )
        result = enrich_screening_with_population(screening, coverage)
        self.assertTrue(result.loc[0, "screening_flag"])
        self.assertTrue(result.loc[0, "coverage_disadvantage"])
        self.assertTrue(result.loc[0, "enhanced_screening_flag"])

    def test_spatial_join_assigns_remoteness_without_overwriting_state(self):
        data = pd.DataFrame(
            {
                "State": pd.Series(["NSW", "VIC", "QLD"], dtype="string"),
                "longitude": [0.5, 1.5, 9.0],
                "latitude": [0.5, 0.5, 9.0],
                "coordinate_valid": [True, True, False],
            }
        )
        boundaries = gpd.GeoDataFrame(
            {
                "RA_CODE21": ["10", "21"],
                "RA_NAME21": [
                    "Major Cities of Australia",
                    "Inner Regional Australia",
                ],
                "STE_NAME21": ["New South Wales", "Victoria"],
                "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            },
            crs="EPSG:4326",
        )
        result = classify_services_by_remoteness(data, boundaries)

        self.assertEqual(result.loc[0, "remoteness_area"], "Major Cities of Australia")
        self.assertEqual(result.loc[1, "remoteness_broad_group"], "Regional Australia")
        self.assertTrue(pd.isna(result.loc[2, "remoteness_area"]))
        self.assertEqual(result.loc[0, "State"], "NSW")


if __name__ == "__main__":
    unittest.main()
