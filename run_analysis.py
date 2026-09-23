"""Run the complete ACECQA exploratory analysis pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import build_analysis_outputs  # noqa: E402
from audit import build_audit_outputs  # noqa: E402
from data import load_and_prepare_data  # noqa: E402
from eda import create_eda_visualisations  # noqa: E402
from interactive import create_interactive_visualisations  # noqa: E402
from population import load_abs_child_population  # noqa: E402
from reporting import write_supporting_reports  # noqa: E402
from spatial import (  # noqa: E402
    classify_services_by_remoteness,
    load_remoteness_boundaries,
)
from visualization import create_visualisations  # noqa: E402
from workflow import write_analysis_workflow  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "Education-services-with-station-access_loc.csv"
REMOTENESS_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "abs"
    / "RA_2021_AUST_GDA2020"
    / "RA_2021_AUST_GDA2020.shp"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHILD_POPULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "abs"
    / "2021_GCP_RA_for_AUS_short-header.zip"
)


def main():
    """Clean the source data, export summaries, and create all figures."""
    print(f"Loading data from {DATA_PATH}")
    data = load_and_prepare_data(DATA_PATH)

    print(f"Classifying services with ABS remoteness boundaries from {REMOTENESS_PATH}")
    remoteness_boundaries = load_remoteness_boundaries(REMOTENESS_PATH)
    data = classify_services_by_remoteness(data, remoteness_boundaries)

    print(f"Loading ABS 2021 population ages 0-13 from {CHILD_POPULATION_PATH}")
    child_population = load_abs_child_population(
        CHILD_POPULATION_PATH, remoteness_boundaries
    )

    print("Creating reproducible data-audit outputs")
    audit_tables = build_audit_outputs(DATA_PATH, data, OUTPUT_DIR)

    print("Building analysis tables and findings")
    tables = build_analysis_outputs(data, OUTPUT_DIR, child_population)
    write_supporting_reports(tables, OUTPUT_DIR)

    print("Creating exploratory data-analysis figures")
    create_eda_visualisations(data, tables, audit_tables, OUTPUT_DIR)

    print("Writing the end-to-end analysis workflow")
    write_analysis_workflow(data, tables, audit_tables, PROJECT_ROOT)

    print("Creating static visualisations")
    create_visualisations(data, tables, remoteness_boundaries, OUTPUT_DIR)

    print("Creating standalone interactive HTML visualisations")
    create_interactive_visualisations(
        data, tables, remoteness_boundaries, OUTPUT_DIR
    )

    print(f"Analysis complete. Results are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
