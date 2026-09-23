# ACECQA geographic service-gap analysis

This project answers the Key Business Question:

> Where do gaps in service quality, accessibility, and operational capacity overlap across Australian education and care services, and which geographic groups warrant closer investigation?

The analytical hierarchy is deliberately geographic:

1. national spatial landscape;
2. service mix as a potential confounder;
3. quality by jurisdiction and remoteness;
4. bus/train proximity by remoteness;
5. approved places per 1,000 children aged 0-13;
6. adjusted quality-accessibility associations;
7. capacity and opening hours by geography;
8. denominator-consistent Centre-Based screening.

Detailed service offerings remain available as a secondary explanatory dimension and in the appendix. They are not the subject of every core chart.

## Project structure

```text
project1/
|-- data/
|   |-- Education-services-with-station-access_loc.csv
|   |-- external/abs/RA_2021_AUST_GDA2020/
|   `-- external/abs/2021_GCP_RA_for_AUS_short-header.zip
|-- scripts/download_abs_boundaries.py
|-- scripts/download_abs_child_population.py
|-- src/
|   |-- config.py          # categories, colours, output paths
|   |-- data.py            # cleaning and derived variables
|   |-- audit.py           # reproducible data-quality audit
|   |-- eda.py             # sample, distribution, and relationship diagnostics
|   |-- spatial.py         # GeoPandas join and equal-area grid
|   |-- analysis.py        # auditable descriptive summaries
|   |-- models.py          # adjusted association models
|   |-- population.py      # ABS 0-13 population and coverage metrics
|   |-- visualization.py   # presentation and appendix figures
|   |-- interactive.py     # focused Plotly explorers
|   |-- reporting.py       # code audit, visual catalogue, sequence
|   `-- workflow.py        # dynamic end-to-end methodology document
|-- tests/test_pipeline.py
|-- run_analysis.py
|-- requirements.txt
|-- ANALYSIS_WORKFLOW.md
`-- outputs/
    |-- audit/
    |-- figures/eda/
    |-- figures/presentation/
    |-- figures/appendix/
    |-- interactive/
    |-- tables/
    |-- data/
    `-- report/
```

## Reproducible preparation

The source CSV is never overwritten and no row is globally deleted merely because it is extreme.

- Ratings remain ordered categories; no arbitrary average quality score is created.
- Missing ratings are excluded only from the relevant quality denominator.
- Negative/missing transport distances are flagged for transport analyses; long valid distances are reviewed and retained.
- Approved capacity is analysed for Centre-Based Care because Family Day Care has no comparable approved-place measure.
- Annual, school-term, and holiday calendars remain separate.
- Overlapping school-term sessions use the union of daily time intervals, preventing double-counted opening hours.
- Coordinates are parsed and joined to official ABS 2021 Remoteness Areas with `geopandas.sjoin`.
- Exact ages 0-13 from ABS Census 2021 G04A are joined by State-specific Remoteness Area code.
- Population coverage reports current-register Centre-Based approved places per 1,000 children and keeps service-count ratios separate.
- Services without valid geography remain available to non-spatial analyses.

Audit outputs document duplicates, missingness, category validity, numeric conversion, capacity/transport reviews, geometry coverage, sample flow, and every cleaning decision.

## Adjusted models

The models estimate associations rather than causal effects. Bus and train are modelled separately.

Quality uses a Binomial GLM with HC1 robust covariance:

```text
Meeting+ ~ log2(distance + 0.1 km) + remoteness + broad service type + state
```

Centre-Based capacity uses log-capacity OLS with HC3 robust covariance:

```text
log(approved places) ~ log2(distance + 0.1 km) + remoteness + state
```

Annual weekly opening hours use HC3 robust OLS:

```text
annual hours ~ log2(distance + 0.1 km) + remoteness + broad service type + state
```

Model coefficients, confidence intervals, sample sizes, formulas, convergence, and fit diagnostics are exported to CSV. The presentation quality figure reports average adjusted predictions over the observed covariate distribution.

## Run

```powershell
cd E:\USYD\6860\project1
python -m pip install -r requirements.txt
python scripts\download_abs_boundaries.py
python scripts\download_abs_child_population.py
python run_analysis.py
python -m unittest discover -s tests -v
```

The ABS boundaries come from the [ASGS Edition 3 Digital Boundary Files](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files). Population comes from the [2021 Census DataPacks](https://www.abs.gov.au/census/find-census-data/datapacks), General Community Profile for Remoteness Areas. Both sources use 2021 RA geography.

## Visual outputs

`outputs/figures/eda/` contains five diagnostic figures used to understand the
data before formal analysis:

1. `E01_data_readiness.png`: completeness and analysis-specific denominators;
2. `E02_univariate_profile.png`: rating, transport, capacity, and hours distributions;
3. `E03_geographic_sample_structure.png`: State x Remoteness sample composition;
4. `E04_raw_relationship_diagnostics.png`: raw distance-capacity/hours relationships.
5. `E05_descriptive_statistics_table.png`: valid n, missingness, central tendency, spread, and robust quantiles.

These figures diagnose data structure and modelling choices. They deliberately
do not duplicate the final policy narrative.

`outputs/figures/presentation/` contains eight core figures:

1. `01_spatial_landscape.png`
2. `02_service_mix_confounder.png`
3. `03_quality_geography.png`
4. `04_accessibility_geography.png`
5. `04b_population_adjusted_coverage.png`
6. `05_adjusted_quality_accessibility.png`
7. `06_operational_geography.png`
8. `07_multidimensional_screening.png`

`outputs/figures/appendix/` contains supporting analyses that are statistically useful but secondary to the main story: detailed offering portfolio, quality by offering, unadjusted transport-distance quintiles, and current-register approval cohorts.

`outputs/interactive/` contains four standalone offline Plotly explorers:

1. `01_service_landscape_explorer.html`
2. `02_quality_geography_explorer.html`
3. `03_population_coverage_explorer.html`
4. `04_geographic_screening_explorer.html`

The service landscape combines ABS remoteness and state outlines with a 50 km
equal-area overview, service-level drill-down, metric switching, click details,
and statistics that recalculate for the visible map extent. The quality explorer
switches between actual Meeting-NQS-or-above rates and within-service-type
national benchmark gaps. The screening explorer combines confidence intervals,
national thresholds, capacity, population coverage, and four screening rules in
one view. The population explorer switches between approved places, Centre-Based
services, and all services per 1,000 children.

The static presentation remains understandable without interaction. The
explorers are reserved for filtering, zoom-sensitive context, hover detail, and
targeted drill-down.

## Interpretation safeguards

- Raw service counts are observed provision; only the dedicated coverage tables use a population denominator.
- The denominator is 2021 Census usual residents aged 0-13, while the service register is a later snapshot; ratios are planning proxies rather than contemporaneous vacancy or unmet-demand estimates.
- Broad ages 0-13 do not perfectly match the eligible population for every service offering, and Census small-area counts are perturbed for confidentiality.
- State x Remoteness cells with fewer than 5,000 children are flagged as small denominators and retained only for cautious screening.
- Public transport distance is a proxy and omits travel time, frequency, affordability, route suitability, car ownership, and actual behaviour.
- Approved places are not enrolments, vacancies, staffing capacity, or waitlists.
- Remoteness describes service location, not family residence or catchment.
- Current-register approval years describe surviving approval cohorts, not historical sector growth.
- Small remote samples produce uncertain rates; rated n and Wilson intervals accompany screening outputs.
- Screening flags identify groups for investigation and are not definitive proof of disadvantage.
- Remoteness Area aggregation can hide local variation within large regions.

See `ANALYSIS_WORKFLOW.md` for the complete data-source-to-decision workflow.
Supporting rationale is also recorded in `outputs/report/code_audit.md`,
`visualisation_catalog.md`, and `key_findings.md`.
