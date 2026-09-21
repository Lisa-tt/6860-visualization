# Final visualisation catalogue

## Presentation figures

### 01 Spatial landscape
- **Question:** Where are approved services observed nationally?
- **Variables:** 75 km equal-area service count, median nearest transport distance, ABS state outlines.
- **Chart:** Projected proportional-symbol map.
- **Why:** Preserves national geography while reducing point overplotting.
- **Interpretation:** Services visibly cluster around population centres, while many inland cells have fewer observed services and longer transport distances.
- **Limitation:** Counts are not adjusted for children or demand and cannot identify unmet need.
- **Placement:** PRESENTATION.

### 02 Service mix as a confounder
- **Question:** Does service composition change across remoteness groups?
- **Variables:** Remoteness, detailed offering prevalence gap from national, service n.
- **Chart:** Zero-centred diverging heatmap.
- **Why:** Direction and magnitude of composition differences are more important than raw counts.
- **Interpretation:** The largest observed shift is Preschool: School-based in Very Remote Australia (+40.7 pp; n=70).
- **Limitation:** Offerings overlap and cells are not mutually exclusive.
- **Placement:** PRESENTATION.

### 03 Geographic quality
- **Question:** Where do overall and Quality Area outcomes differ geographically?
- **Variables:** Overall rating composition by state/remoteness; Meeting+ gaps for Overall and QA1-QA7.
- **Chart:** 100% stacked bars plus zero-centred benchmark heatmap with n.
- **Why:** Composition retains the ordinal categories; benchmark gaps reveal systematic differences.
- **Interpretation:** NT has the highest state below-NQS share (21.9%); Very Remote Australia is highest by remoteness (30.6%).
- **Limitation:** Raw geographic differences may still reflect service mix and other unmeasured factors.
- **Placement:** PRESENTATION.

### 04 Geographic accessibility
- **Question:** How does proximity to bus and train infrastructure vary with remoteness?
- **Variables:** Bus/train median and P90 distance by remoteness.
- **Chart:** Two median-to-P90 distance profiles on a log scale.
- **Why:** The trajectories show both the typical access shift and the long-distance tail without repeating a dot-and-whisker design.
- **Interpretation:** Median nearest-mode distance increases from 1.89 km in Major Cities to 230.03 km in Very Remote Australia.
- **Limitation:** Distance is not travel time, service frequency, affordability, route suitability, or actual family travel behaviour.
- **Placement:** PRESENTATION.

### 04b Population-adjusted coverage
- **Question:** Does observed Centre-Based supply keep pace with the number of children aged 0-13?
- **Variables:** ABS 2021 population aged 0-13, Centre-Based approved places, State-specific Remoteness Area.
- **Chart:** Choropleth plus State x Remoteness benchmark heatmap.
- **Why:** Converts raw provision into a compatible places-per-child indicator while preserving geography.
- **Interpretation:** Approved places per 1,000 children decline from 293.9 in Major Cities to 141.2 in Very Remote Australia.
- **Limitation:** The 2021 Census denominator and current service snapshot are not contemporaneous; approved places are not vacancies or demand.
- **Placement:** PRESENTATION and INTERACTIVE EXPLORER.

### 05 Adjusted quality-accessibility association
- **Question:** Does transport distance remain associated with Meeting+ after geographic and broad-type adjustment?
- **Variables:** Meeting+, log2 distance, remoteness, state, broad service type.
- **Chart:** Adjusted marginal probability curves with 95% bands and an unadjusted quintile trend.
- **Why:** Directly shows how adjustment changes the apparent relationship.
- **Interpretation:** Per distance doubling, adjusted ORs are 0.965 for bus and 1.003 for train; the raw gradient is substantially attenuated.
- **Limitation:** Observational models address measured composition only and do not estimate causal effects.
- **Placement:** PRESENTATION.

### 06 Operational geography
- **Question:** Do capacity and opening hours amplify geographic access gaps?
- **Variables:** Capacity/hours by remoteness and adjusted bus/train effects.
- **Chart:** Capacity bars, an opening-hours dumbbell, and adjusted-effect bars with printed robust intervals.
- **Why:** Separates the operational levels from model evidence while exposing the very small remote Family Day Care samples.
- **Interpretation:** The adjusted Centre-Based capacity estimate is -1.64% per bus-distance doubling; opening-hours estimates are reported separately rather than inferred from service offering.
- **Limitation:** Approved places do not measure enrolment, vacancies, staffing, or waitlists; hours use annual calendars only.
- **Placement:** PRESENTATION.

### 07 Multidimensional screening
- **Question:** Which Centre-Based State x Remoteness groups combine weaker quality, poorer proximity, smaller facilities, and lower population-adjusted supply?
- **Variables:** Median transport distance, below-NQS rate, median capacity, approved places per 1,000 children, rated n.
- **Chart:** Four aligned deviation-bar panels for quality, transport, facility size, and population coverage.
- **Why:** Profiles every flagged group against four transparent rules without creating an arbitrary weighted score.
- **Interpretation:** 12 eligible groups meet all four national-benchmark screening rules.
- **Limitation:** Flags warrant further investigation; they are not proof of disadvantage, and small groups remain uncertain.
- **Placement:** PRESENTATION; the full comparison and uncertainty remain in the INTERACTIVE EXPLORER.

## Appendix figures

- **A01 Detailed offering portfolio:** documents overlapping offerings and combinations; APPENDIX.
- **A02 Quality by offering:** Wilson intervals and rating n; APPENDIX because offering is secondary.
- **A03 Unadjusted distance quintiles:** exploratory comparison retained to show confounding; APPENDIX.
- **A04 Approval cohorts:** bars for approval-year composition of current services; APPENDIX and not historical growth.

## Interactive explorers

- **Service landscape explorer:** state, remoteness, and broad-type filtered geographic drill-down.
- **Quality geography explorer:** broad-type filters with state x QA benchmark hover n.
- **Population coverage explorer:** linked State filter and metric switch for places or services per 1,000 children.
- **Centre-Based screening explorer:** hover detail for all four compatible-denominator synthesis dimensions and uncertainty.
