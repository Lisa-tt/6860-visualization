"""Adjusted association models for quality and operational outcomes."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.special import expit

from config import REMOTENESS_ORDER, STATE_ORDER


DISTANCE_MODES = {
    "Bus": "DistanceToBusStation_km",
    "Train": "DistanceToTrainStation_km",
}

DISTANCE_OFFSET_KM = 0.1


def _model_categories(data):
    """Convert pandas StringDtype columns for Patsy 0.5 compatibility."""
    result = data.copy()
    for column in ["State", "remoteness_area", "ServiceType"]:
        if column in result:
            result[column] = result[column].astype(object)
    return result


def _distance_grid(values, points=40):
    lower = max(0.01, float(values.quantile(0.01)))
    upper = float(values.quantile(0.99))
    return np.geomspace(lower, upper, points)


def _average_logistic_predictions(result, sample, distance_values):
    """Average predictions over the observed covariate distribution."""
    design_info = result.model.data.design_info
    parameters = result.params.to_numpy()
    covariance = result.cov_params().to_numpy()
    records = []
    for distance in distance_values:
        scenario = sample.copy()
        scenario["log_distance"] = np.log2(distance + DISTANCE_OFFSET_KM)
        design = np.asarray(
            build_design_matrices([design_info], scenario)[0], dtype=float
        )
        probability = expit(design @ parameters)
        estimate = float(probability.mean())
        gradient = (design * (probability * (1 - probability))[:, None]).mean(axis=0)
        standard_error = float(np.sqrt(gradient @ covariance @ gradient))
        records.append(
            {
                "distance_km": distance,
                "adjusted_meeting_or_above_pct": 100 * estimate,
                "ci95_lower_pct": 100 * max(0, estimate - 1.96 * standard_error),
                "ci95_upper_pct": 100 * min(1, estimate + 1.96 * standard_error),
            }
        )
    return pd.DataFrame(records)


def build_adjusted_quality_models(data):
    """Fit separate bus and train quality models with geographic controls."""
    sample = data[
        data["is_rated"]
        & data["State"].isin(STATE_ORDER)
        & data["remoteness_area"].isin(REMOTENESS_ORDER)
        & data["ServiceType"].notna()
    ].copy()
    sample = _model_categories(sample)
    sample["meeting_binary"] = sample["is_meeting_or_above"].astype(int)
    coefficients = []
    predictions = []
    diagnostics = []
    formula = (
        "meeting_binary ~ log_distance + C(remoteness_area) "
        "+ C(ServiceType) + C(State)"
    )
    for mode, distance_column in DISTANCE_MODES.items():
        model_data = sample.dropna(subset=[distance_column]).copy()
        model_data = model_data[model_data[distance_column].ge(0)].copy()
        model_data["log_distance"] = np.log2(
            model_data[distance_column] + DISTANCE_OFFSET_KM
        )
        result = smf.glm(
            formula,
            data=model_data,
            family=sm.families.Binomial(),
        ).fit(cov_type="HC1")
        interval = result.conf_int().loc["log_distance"]
        coefficients.append(
            {
                "model": "Quality Meeting+",
                "transport_mode": mode,
                "services": int(result.nobs),
                "coefficient_log_odds": result.params["log_distance"],
                "robust_standard_error": result.bse["log_distance"],
                "p_value": result.pvalues["log_distance"],
                "odds_ratio_per_distance_doubling": np.exp(
                    result.params["log_distance"]
                ),
                "or_ci95_lower": np.exp(interval.iloc[0]),
                "or_ci95_upper": np.exp(interval.iloc[1]),
                "distance_offset_km": DISTANCE_OFFSET_KM,
            }
        )
        prediction = _average_logistic_predictions(
            result, model_data, _distance_grid(model_data[distance_column])
        )
        prediction.insert(0, "transport_mode", mode)
        prediction["model_services"] = int(result.nobs)
        predictions.append(prediction)
        diagnostics.append(
            {
                "model": "Quality Meeting+",
                "transport_mode": mode,
                "family": "Binomial GLM with HC1 robust covariance",
                "formula": formula,
                "services": int(result.nobs),
                "outcome_events": int(model_data["meeting_binary"].sum()),
                "converged": bool(result.converged),
                "fit_statistic": "McFadden pseudo-R2",
                "fit_value": 1 - result.llf / result.llnull,
            }
        )
    return (
        pd.DataFrame(coefficients),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def _fit_operational_model(
    model_data,
    formula,
    outcome,
    mode,
    effect_transform,
    effect_unit,
):
    result = smf.ols(formula, data=model_data).fit(cov_type="HC3")
    interval = result.conf_int().loc["log_distance"]
    estimate = effect_transform(result.params["log_distance"])
    lower = effect_transform(interval.iloc[0])
    upper = effect_transform(interval.iloc[1])
    effect = {
        "outcome": outcome,
        "transport_mode": mode,
        "services": int(result.nobs),
        "adjusted_effect_per_distance_doubling": estimate,
        "ci95_lower": min(lower, upper),
        "ci95_upper": max(lower, upper),
        "effect_unit": effect_unit,
        "p_value": result.pvalues["log_distance"],
        "distance_offset_km": DISTANCE_OFFSET_KM,
    }
    diagnostic = {
        "model": outcome,
        "transport_mode": mode,
        "family": "OLS with HC3 robust covariance",
        "formula": formula,
        "services": int(result.nobs),
        "outcome_events": np.nan,
        "converged": True,
        "fit_statistic": "Adjusted R-squared",
        "fit_value": result.rsquared_adj,
    }
    return effect, diagnostic


def build_adjusted_operational_models(data):
    """Test transport associations with capacity and annual opening hours."""
    centre = data[
        data["ServiceType"].eq("Centre-Based Care")
        & data["NumberOfApprovedPlaces"].gt(0)
        & data["State"].isin(STATE_ORDER)
        & data["remoteness_area"].isin(REMOTENESS_ORDER)
    ].copy()
    centre = _model_categories(centre)
    centre["log_capacity"] = np.log(centre["NumberOfApprovedPlaces"])

    hours = data[
        data["annual_hours_valid"]
        & data["State"].isin(STATE_ORDER)
        & data["remoteness_area"].isin(REMOTENESS_ORDER)
        & data["ServiceType"].notna()
    ].copy()
    hours = _model_categories(hours)

    effects = []
    diagnostics = []
    for mode, distance_column in DISTANCE_MODES.items():
        capacity_data = centre.dropna(subset=[distance_column]).copy()
        capacity_data["log_distance"] = np.log2(
            capacity_data[distance_column] + DISTANCE_OFFSET_KM
        )
        effect, diagnostic = _fit_operational_model(
            capacity_data,
            "log_capacity ~ log_distance + C(remoteness_area) + C(State)",
            "Centre-Based approved capacity",
            mode,
            lambda coefficient: 100 * (np.exp(coefficient) - 1),
            "% change in approved places",
        )
        effects.append(effect)
        diagnostics.append(diagnostic)

        hours_data = hours.dropna(subset=[distance_column]).copy()
        hours_data["log_distance"] = np.log2(
            hours_data[distance_column] + DISTANCE_OFFSET_KM
        )
        effect, diagnostic = _fit_operational_model(
            hours_data,
            "annual_weekly_operating_hours ~ log_distance + C(remoteness_area) "
            "+ C(ServiceType) + C(State)",
            "Annual weekly opening hours",
            mode,
            lambda coefficient: coefficient,
            "hours per week",
        )
        effects.append(effect)
        diagnostics.append(diagnostic)
    return pd.DataFrame(effects), pd.DataFrame(diagnostics)


def build_model_outputs(data):
    quality_coefficients, quality_predictions, quality_diagnostics = (
        build_adjusted_quality_models(data)
    )
    operational_effects, operational_diagnostics = (
        build_adjusted_operational_models(data)
    )
    diagnostics = pd.concat(
        [quality_diagnostics, operational_diagnostics], ignore_index=True
    )
    return {
        "quality_model_coefficients": quality_coefficients,
        "quality_adjusted_predictions": quality_predictions,
        "operational_model_effects": operational_effects,
        "model_diagnostics": diagnostics,
    }
