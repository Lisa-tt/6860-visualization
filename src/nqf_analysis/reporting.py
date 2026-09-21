"""Write audit decisions, visual metadata, and presentation guidance."""

from pathlib import Path

from .config import output_paths


def _write(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_code_audit(report_dir):
    lines = [
        "# Existing-code audit and targeted refactor decisions",
        "",
        "The refactor preserves statistically sound preparation and summaries while changing the analytical hierarchy from service offering to geography.",
        "",
        "| Existing section or visual | Decision | Reason |",
        "|---|---|---|",
        "| Text, numeric, date, rating, and coordinate preparation | KEEP / MODIFY | Core derivations are reusable; add explicit audit flags and overlap-safe school-term hours. |",
        "| GeoPandas ABS Remoteness Area spatial join | KEEP | Correct spatial method, CRS handling, match diagnostics, and source-state preservation. |",
        "| Wilson confidence intervals | KEEP | Appropriate uncertainty interval for proportions, especially smaller groups. |",
        "| Detailed offering portfolio | MOVE TO APPENDIX | Multi-label logic is correct, but offering is not the primary business question. |",
        "| Service mix by remoteness | KEEP | One methodological visual demonstrates geographic composition as a confounder. |",
        "| State quality composition and national QA benchmark | KEEP / MODIFY | Retain 100% composition and zero-centred gaps; replace the broad-type panel with remoteness. |",
        "| Quality by detailed offering | MOVE TO APPENDIX | Statistically sound but repetitive in the core narrative. |",
        "| Accessibility by remoteness and offering | MODIFY | Replace repeated dot ranges with bus/train median-to-P90 profiles by remoteness; remove offering panel. |",
        "| Quality by distance quintile | MOVE TO APPENDIX | Useful unadjusted exploration but confounded by geography and service composition. |",
        "| Operations by detailed offering | REPLACE | Core question is whether capacity and hours amplify geographic gaps. |",
        "| Approval-year line | REPLACE | Current-register cohorts require bars and must not imply historical sector growth. |",
        "| Mixed-denominator synthesis | REPLACE | Restrict every encoded dimension to Centre-Based Care and call outputs screening flags. |",
        "| Interactive service map | KEEP / MODIFY | Retain geographic exploration; replace offering-heavy filters with state, remoteness, and broad type. |",
        "| Interactive offering quality and compound explorers | MODIFY | Use broad type only for quality and a single denominator-consistent Centre-Based screening view. |",
        "",
        "## Preserved statistical decisions",
        "",
        "- Ordinal ratings remain categorical; no arbitrary average rating score is calculated.",
        "- Percentage compositions use recognised ratings as their explicit denominator.",
        "- Skewed transport distances use robust quantiles and log display scales; presentation profiles pair the median with P90 while tables retain quartiles.",
        "- Bus and train remain separate because infrastructure availability differs structurally.",
        "- Extreme remote distances are reviewed and retained rather than deleted by an IQR rule.",
        "- Annual, school-term, and holiday calendars remain separate.",
        "- Population-adjusted coverage uses exact ages 0-13 from ABS 2021 GCP at State-specific RA codes; temporal and service-age limitations are stated explicitly.",
    ]
    _write(Path(report_dir) / "code_audit.md", lines)


def write_visualisation_catalog(tables, report_dir):
    state = tables["state_summary"]
    remote = tables["remoteness_summary"]
    transport = tables["transport_by_remoteness"]
    mix = tables["offering_by_remoteness"]
    quality_models = tables["quality_model_coefficients"]
    operational_models = tables["operational_model_effects"]
    compound = tables["compound_disadvantage"]
    coverage = tables["population_coverage_by_remoteness"]

    highest_state = state.loc[state["below_nqs_pct_of_rated"].idxmax()]
    highest_remote = remote.loc[remote["below_nqs_pct_of_rated"].idxmax()]
    mix_shift = mix.loc[mix["gap_from_national_pp"].abs().idxmax()]
    city_distance = transport[
        transport["remoteness_area"].eq("Major Cities of Australia")
        & transport["transport_mode"].eq("Nearest listed mode")
    ].iloc[0]
    remote_distance = transport[
        transport["remoteness_area"].eq("Very Remote Australia")
        & transport["transport_mode"].eq("Nearest listed mode")
    ].iloc[0]
    bus_quality = quality_models[quality_models["transport_mode"].eq("Bus")].iloc[0]
    train_quality = quality_models[quality_models["transport_mode"].eq("Train")].iloc[0]
    capacity_bus = operational_models[
        operational_models["outcome"].eq("Centre-Based approved capacity")
        & operational_models["transport_mode"].eq("Bus")
    ].iloc[0]
    flags = int(compound["enhanced_screening_flag"].sum())
    city_coverage = coverage[
        coverage["remoteness_area"].eq("Major Cities of Australia")
    ].iloc[0]
    very_remote_coverage = coverage[
        coverage["remoteness_area"].eq("Very Remote Australia")
    ].iloc[0]

    lines = [
        "# Final visualisation catalogue",
        "",
        "## Presentation figures",
        "",
        "### 01 Spatial landscape",
        "- **Question:** Where are approved services observed nationally?",
        "- **Variables:** 75 km equal-area service count, median nearest transport distance, ABS state outlines.",
        "- **Chart:** Projected proportional-symbol map.",
        "- **Why:** Preserves national geography while reducing point overplotting.",
        "- **Interpretation:** Services visibly cluster around population centres, while many inland cells have fewer observed services and longer transport distances.",
        "- **Limitation:** Counts are not adjusted for children or demand and cannot identify unmet need.",
        "- **Placement:** PRESENTATION.",
        "",
        "### 02 Service mix as a confounder",
        "- **Question:** Does service composition change across remoteness groups?",
        "- **Variables:** Remoteness, detailed offering prevalence gap from national, service n.",
        "- **Chart:** Zero-centred diverging heatmap.",
        "- **Why:** Direction and magnitude of composition differences are more important than raw counts.",
        f"- **Interpretation:** The largest observed shift is {mix_shift['offering']} in {mix_shift['remoteness_area']} ({mix_shift['gap_from_national_pp']:+.1f} pp; n={int(mix_shift['services']):,}).",
        "- **Limitation:** Offerings overlap and cells are not mutually exclusive.",
        "- **Placement:** PRESENTATION.",
        "",
        "### 03 Geographic quality",
        "- **Question:** Where do overall and Quality Area outcomes differ geographically?",
        "- **Variables:** Overall rating composition by state/remoteness; Meeting+ gaps for Overall and QA1-QA7.",
        "- **Chart:** 100% stacked bars plus zero-centred benchmark heatmap with n.",
        "- **Why:** Composition retains the ordinal categories; benchmark gaps reveal systematic differences.",
        f"- **Interpretation:** {highest_state['state']} has the highest state below-NQS share ({highest_state['below_nqs_pct_of_rated']:.1f}%); {highest_remote['remoteness_area']} is highest by remoteness ({highest_remote['below_nqs_pct_of_rated']:.1f}%).",
        "- **Limitation:** Raw geographic differences may still reflect service mix and other unmeasured factors.",
        "- **Placement:** PRESENTATION.",
        "",
        "### 04 Geographic accessibility",
        "- **Question:** How does proximity to bus and train infrastructure vary with remoteness?",
        "- **Variables:** Bus/train median and P90 distance by remoteness.",
        "- **Chart:** Two median-to-P90 distance profiles on a log scale.",
        "- **Why:** The trajectories show both the typical access shift and the long-distance tail without repeating a dot-and-whisker design.",
        f"- **Interpretation:** Median nearest-mode distance increases from {city_distance['median_distance_km']:.2f} km in Major Cities to {remote_distance['median_distance_km']:.2f} km in Very Remote Australia.",
        "- **Limitation:** Distance is not travel time, service frequency, affordability, route suitability, or actual family travel behaviour.",
        "- **Placement:** PRESENTATION.",
        "",
        "### 04b Population-adjusted coverage",
        "- **Question:** Does observed Centre-Based supply keep pace with the number of children aged 0-13?",
        "- **Variables:** ABS 2021 population aged 0-13, Centre-Based approved places, State-specific Remoteness Area.",
        "- **Chart:** Choropleth plus State x Remoteness benchmark heatmap.",
        "- **Why:** Converts raw provision into a compatible places-per-child indicator while preserving geography.",
        f"- **Interpretation:** Approved places per 1,000 children decline from {city_coverage['approved_places_per_1000_children']:.1f} in Major Cities to {very_remote_coverage['approved_places_per_1000_children']:.1f} in Very Remote Australia.",
        "- **Limitation:** The 2021 Census denominator and current service snapshot are not contemporaneous; approved places are not vacancies or demand.",
        "- **Placement:** PRESENTATION and INTERACTIVE EXPLORER.",
        "",
        "### 05 Adjusted quality-accessibility association",
        "- **Question:** Does transport distance remain associated with Meeting+ after geographic and broad-type adjustment?",
        "- **Variables:** Meeting+, log2 distance, remoteness, state, broad service type.",
        "- **Chart:** Adjusted marginal probability curves with 95% bands and an unadjusted quintile trend.",
        "- **Why:** Directly shows how adjustment changes the apparent relationship.",
        f"- **Interpretation:** Per distance doubling, adjusted ORs are {bus_quality['odds_ratio_per_distance_doubling']:.3f} for bus and {train_quality['odds_ratio_per_distance_doubling']:.3f} for train; the raw gradient is substantially attenuated.",
        "- **Limitation:** Observational models address measured composition only and do not estimate causal effects.",
        "- **Placement:** PRESENTATION.",
        "",
        "### 06 Operational geography",
        "- **Question:** Do capacity and opening hours amplify geographic access gaps?",
        "- **Variables:** Capacity/hours by remoteness and adjusted bus/train effects.",
        "- **Chart:** Capacity bars, an opening-hours dumbbell, and adjusted-effect bars with printed robust intervals.",
        "- **Why:** Separates the operational levels from model evidence while exposing the very small remote Family Day Care samples.",
        f"- **Interpretation:** The adjusted Centre-Based capacity estimate is {capacity_bus['adjusted_effect_per_distance_doubling']:.2f}% per bus-distance doubling; opening-hours estimates are reported separately rather than inferred from service offering.",
        "- **Limitation:** Approved places do not measure enrolment, vacancies, staffing, or waitlists; hours use annual calendars only.",
        "- **Placement:** PRESENTATION.",
        "",
        "### 07 Multidimensional screening",
        "- **Question:** Which Centre-Based State x Remoteness groups combine weaker quality, poorer proximity, smaller facilities, and lower population-adjusted supply?",
        "- **Variables:** Median transport distance, below-NQS rate, median capacity, approved places per 1,000 children, rated n.",
        "- **Chart:** Four aligned deviation-bar panels for quality, transport, facility size, and population coverage.",
        "- **Why:** Profiles every flagged group against four transparent rules without creating an arbitrary weighted score.",
        f"- **Interpretation:** {flags} eligible groups meet all four national-benchmark screening rules.",
        "- **Limitation:** Flags warrant further investigation; they are not proof of disadvantage, and small groups remain uncertain.",
        "- **Placement:** PRESENTATION; the full comparison and uncertainty remain in the INTERACTIVE EXPLORER.",
        "",
        "## Appendix figures",
        "",
        "- **A01 Detailed offering portfolio:** documents overlapping offerings and combinations; APPENDIX.",
        "- **A02 Quality by offering:** Wilson intervals and rating n; APPENDIX because offering is secondary.",
        "- **A03 Unadjusted distance quintiles:** exploratory comparison retained to show confounding; APPENDIX.",
        "- **A04 Approval cohorts:** bars for approval-year composition of current services; APPENDIX and not historical growth.",
        "",
        "## Interactive explorers",
        "",
        "- **Service landscape explorer:** state, remoteness, and broad-type filtered geographic drill-down.",
        "- **Quality geography explorer:** broad-type filters with state x QA benchmark hover n.",
        "- **Population coverage explorer:** linked State filter and metric switch for places or services per 1,000 children.",
        "- **Centre-Based screening explorer:** hover detail for all four compatible-denominator synthesis dimensions and uncertainty.",
    ]
    _write(Path(report_dir) / "visualisation_catalog.md", lines)


def write_presentation_sequence(report_dir):
    lines = [
        "# Recommended 5-10 minute presentation sequence",
        "",
        "1. **0:00-0:40 | KBQ and scope** - Where do quality, accessibility, population-adjusted coverage, and operational gaps overlap? State that the register is cross-sectional.",
        "2. **0:40-1:35 | Spatial landscape** - Establish where services are observed; avoid the word underserved.",
        "3. **1:35-2:20 | Service mix as confounder** - One slide explaining why raw geographic comparisons need adjustment.",
        "4. **2:20-3:35 | Geographic quality** - Show state/remoteness compositions and systematic QA benchmark gaps.",
        "5. **3:35-4:15 | Transport accessibility** - Show the scale and skew of bus/train differences by remoteness.",
        "6. **4:15-5:00 | Population-adjusted coverage** - Contrast service counts with approved places per 1,000 children and state the 2021/current-snapshot mismatch.",
        "7. **5:00-6:10 | Adjusted quality-accessibility model** - Contrast raw quintiles with adjusted marginal predictions; describe association, not causation.",
        "8. **6:10-7:10 | Operational geography** - Explain capacity denominator, hours basis, and adjusted effects.",
        "9. **7:10-8:20 | Multidimensional screening** - Identify groups warranting investigation and state all four transparent rules.",
        "10. **8:20-9:00 | Implications and limitations** - Recommend targeted validation, a contemporaneous age-specific denominator, and local context review rather than direct resource allocation from this snapshot alone.",
        "",
        "For a five-minute version, move the service-mix figure to a brief methodological callout and summarise operations verbally before the synthesis.",
    ]
    _write(Path(report_dir) / "presentation_sequence.md", lines)


def write_supporting_reports(tables, output_dir):
    report_dir = Path(output_paths(output_dir)["report"])
    write_code_audit(report_dir)
    write_visualisation_catalog(tables, report_dir)
    write_presentation_sequence(report_dir)
