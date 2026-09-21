"""Shared definitions used by the analysis and visualisations."""

from pathlib import Path


RATING_ORDER = [
    "Significant Improvement Required",
    "Working Towards NQS",
    "Meeting NQS",
    "Exceeding NQS",
    "Excellent",
]

LOW_RATINGS = {
    "Significant Improvement Required",
    "Working Towards NQS",
}

HIGH_RATINGS = {
    "Exceeding NQS",
    "Excellent",
}

RATING_GROUP_ORDER = [
    "No current rating",
    "Below NQS",
    "Meets NQS",
    "Above NQS",
]

RATING_COLORS = {
    "Significant Improvement Required": "#B91C1C",
    "Working Towards NQS": "#E76F51",
    "Meeting NQS": "#2A9D8F",
    "Exceeding NQS": "#457B9D",
    "Excellent": "#F4A261",
    "No current rating": "#9CA3AF",
}

RATING_GROUP_COLORS = {
    "No current rating": "#9CA3AF",
    "Below NQS": "#D1495B",
    "Meets NQS": "#2A9D8F",
    "Above NQS": "#457B9D",
}

QUALITY_AREA_COLUMNS = ["QualityArea{}Rating".format(i) for i in range(1, 8)]
QUALITY_AREA_LABELS = {
    "QualityArea1Rating": "QA1",
    "QualityArea2Rating": "QA2",
    "QualityArea3Rating": "QA3",
    "QualityArea4Rating": "QA4",
    "QualityArea5Rating": "QA5",
    "QualityArea6Rating": "QA6",
    "QualityArea7Rating": "QA7",
}

QUALITY_AREA_NAMES = {
    "QA1": "Educational program and practice",
    "QA2": "Children's health and safety",
    "QA3": "Physical environment",
    "QA4": "Staffing arrangements",
    "QA5": "Relationships with children",
    "QA6": "Collaborative partnerships",
    "QA7": "Governance and leadership",
}

DETAILED_SERVICE_OFFERINGS = {
    "Long Day Care": "Long Day Care",
    "Preschool/Kindergarten - Part of a School": "Preschool: School-based",
    "Preschool/Kindergarten - Stand alone": "Preschool: Stand-alone",
    "Outside school Hours Care - After School": "OSHC: After School",
    "Outside school Hours Care - Before School": "OSHC: Before School",
    "Outside school Hours Care - Vacation Care": "OSHC: Vacation Care",
    "Other": "Other",
}

OFFERING_ORDER = list(DETAILED_SERVICE_OFFERINGS.values())

OFFERING_COLORS = {
    "Long Day Care": "#2A9D8F",
    "Preschool: School-based": "#7C3AED",
    "Preschool: Stand-alone": "#A855F7",
    "OSHC: After School": "#457B9D",
    "OSHC: Before School": "#60A5FA",
    "OSHC: Vacation Care": "#F4A261",
    "Other": "#9CA3AF",
}

STATE_ORDER = ["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"]

STATE_NAME_TO_ABBR = {
    "New South Wales": "NSW",
    "Victoria": "VIC",
    "Queensland": "QLD",
    "South Australia": "SA",
    "Western Australia": "WA",
    "Tasmania": "TAS",
    "Northern Territory": "NT",
    "Australian Capital Territory": "ACT",
    "Other Territories": "OT",
}

REMOTENESS_ORDER = [
    "Major Cities of Australia",
    "Inner Regional Australia",
    "Outer Regional Australia",
    "Remote Australia",
    "Very Remote Australia",
]

REMOTENESS_COLORS = {
    "Major Cities of Australia": "#4C78A8",
    "Inner Regional Australia": "#72B7B2",
    "Outer Regional Australia": "#F2CF5B",
    "Remote Australia": "#F28E2B",
    "Very Remote Australia": "#D1495B",
}

BROAD_REMOTENESS_ORDER = ["Major Cities", "Regional Australia", "Remote Australia"]
BROAD_REMOTENESS_MAP = {
    "Major Cities of Australia": "Major Cities",
    "Inner Regional Australia": "Regional Australia",
    "Outer Regional Australia": "Regional Australia",
    "Remote Australia": "Remote Australia",
    "Very Remote Australia": "Remote Australia",
}

TRANSPORT_BAND_ORDER = [
    "0-1 km",
    "1-2 km",
    "2-5 km",
    "5-10 km",
    "Over 10 km",
]

CAPACITY_BAND_ORDER = [
    "Up to 40 places",
    "41-80 places",
    "81-120 places",
    "Over 120 places",
]


def output_paths(output_dir):
    """Return and create the standard output directories."""
    output_dir = Path(output_dir)
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "eda_figures": output_dir / "figures" / "eda",
        "presentation_figures": output_dir / "figures" / "presentation",
        "appendix_figures": output_dir / "figures" / "appendix",
        "interactive": output_dir / "interactive",
        "tables": output_dir / "tables",
        "data": output_dir / "data",
        "audit": output_dir / "audit",
        "report": output_dir / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
