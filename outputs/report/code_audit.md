# Existing-code audit and targeted refactor decisions

The refactor preserves statistically sound preparation and summaries while changing the analytical hierarchy from service offering to geography.

| Existing section or visual | Decision | Reason |
|---|---|---|
| Text, numeric, date, rating, and coordinate preparation | KEEP / MODIFY | Core derivations are reusable; add explicit audit flags and overlap-safe school-term hours. |
| GeoPandas ABS Remoteness Area spatial join | KEEP | Correct spatial method, CRS handling, match diagnostics, and source-state preservation. |
| Wilson confidence intervals | KEEP | Appropriate uncertainty interval for proportions, especially smaller groups. |
| Detailed offering portfolio | MOVE TO APPENDIX | Multi-label logic is correct, but offering is not the primary business question. |
| Service mix by remoteness | KEEP | One methodological visual demonstrates geographic composition as a confounder. |
| State quality composition and national QA benchmark | KEEP / MODIFY | Retain 100% composition and zero-centred gaps; replace the broad-type panel with remoteness. |
| Quality by detailed offering | MOVE TO APPENDIX | Statistically sound but repetitive in the core narrative. |
| Accessibility by remoteness and offering | MODIFY | Replace repeated dot ranges with bus/train median-to-P90 profiles by remoteness; remove offering panel. |
| Quality by distance quintile | MOVE TO APPENDIX | Useful unadjusted exploration but confounded by geography and service composition. |
| Operations by detailed offering | REPLACE | Core question is whether capacity and hours amplify geographic gaps. |
| Approval-year line | REPLACE | Current-register cohorts require bars and must not imply historical sector growth. |
| Mixed-denominator synthesis | REPLACE | Restrict every encoded dimension to Centre-Based Care and call outputs screening flags. |
| Interactive service map | KEEP / MODIFY | Retain geographic exploration; replace offering-heavy filters with state, remoteness, and broad type. |
| Interactive offering quality and compound explorers | MODIFY | Use broad type only for quality and a single denominator-consistent Centre-Based screening view. |

## Preserved statistical decisions

- Ordinal ratings remain categorical; no arbitrary average rating score is calculated.
- Percentage compositions use recognised ratings as their explicit denominator.
- Skewed transport distances use robust quantiles and log display scales; presentation profiles pair the median with P90 while tables retain quartiles.
- Bus and train remain separate because infrastructure availability differs structurally.
- Extreme remote distances are reviewed and retained rather than deleted by an IQR rule.
- Annual, school-term, and holiday calendars remain separate.
- Population-adjusted coverage uses exact ages 0-13 from ABS 2021 GCP at State-specific RA codes; temporal and service-age limitations are stated explicitly.
