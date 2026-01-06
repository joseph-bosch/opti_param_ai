# optimizer_service.py
"""
Optimizer service for recommending machine parameters.

"""

from __future__ import annotations

import math
import optuna
import pandas as pd
from typing import Dict, Any, Tuple, List

from .model_service import POINTS
from .specs_service import get_feature_bounds



REC_PREFIX = "Rec_"
FE_PREFIX = "FE_"
ENV_LOCKED = {"Temperature", "Humidity", "Valve_Filter_Status"} 
FE_TIME_INDEX_DEFAULT = 0.0  


def _to_float_safe(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)

        return float(str(x).strip())
    except Exception:
        return 0.0


def _sum_keys(d: Dict[str, Any], must_contain: List[str], letter: str | None = None) -> float:
    """Sum over Rec_* fields matching all substrings in must_contain; optionally restrict to a letter (A..F)."""
    total = 0.0
    for k, v in d.items():
        if not k.startswith(REC_PREFIX):
            continue
        if letter is not None and not k.startswith(f"{REC_PREFIX}{letter}_"):
            continue
        if all(s in k for s in must_contain):
            total += _to_float_safe(v)
    return total


def _derive_engineered_features(all_feats: Dict[str, Any]) -> Dict[str, float]:
    """
    Recompute engineered FE_* features from candidate + fixed context.

    Matches the training engineered set you had:
      - FE_All_Total_Voltage  = sum of all Rec_*_*_Voltage
      - FE_All_Total_Powder   = sum of all Rec_*_*_Powder
      - FE_C_Total_Voltage    = sum of Rec_C_*_Voltage
      - FE_D_Total_Voltage    = sum of Rec_D_*_Voltage
      - FE_Time_Index         = default 0.0 unless caller provided one

    If any of these don't exist in the training preprocessor, model_service will drop them safely.
    """
    fe: Dict[str, float] = {}


    fe["FE_All_Total_Voltage"] = _sum_keys(all_feats, ["_Voltage"])
    fe["FE_All_Total_Powder"] = _sum_keys(all_feats, ["_Powder"])


    fe["FE_C_Total_Voltage"] = _sum_keys(all_feats, ["_Voltage"], letter="C")
    fe["FE_D_Total_Voltage"] = _sum_keys(all_feats, ["_Voltage"], letter="D")


    if "FE_Time_Index" in all_feats:
        try:
            fe["FE_Time_Index"] = _to_float_safe(all_feats["FE_Time_Index"])
        except Exception:
            fe["FE_Time_Index"] = FE_TIME_INDEX_DEFAULT
    else:
        fe["FE_Time_Index"] = FE_TIME_INDEX_DEFAULT

    return fe


# ----------------------------------------------------------
# Helper: Compute local bounds around current value
# ----------------------------------------------------------
def _local_bounds(meta: dict, current_val: float | None, step_pct: float) -> Tuple[float, float]:
    lo, hi = float(meta["lo"]), float(meta["hi"])


    if current_val is None or not (lo <= current_val <= hi):
        return lo, hi


    span = max(hi - lo, 1e-12)
    lb = max(lo, current_val - step_pct * span)
    ub = min(hi, current_val + step_pct * span)

    if lb >= ub:

        return lo, hi

    return lb, ub


# ----------------------------------------------------------
# Build Optuna objective
# ----------------------------------------------------------
def make_objective(model_service,
                   type_code: str,
                   specs: dict,
                   fixed_context: dict,
                   current: dict | None,
                   step_pct: float):
    """
    Creates the Optuna objective function used for parameter search.

    """

    fb = get_feature_bounds(type_code) or {}
    if not isinstance(fb, dict):
        fb = {}


    tunable: Dict[str, dict] = {
        k: v for k, v in fb.items()
        if v.get("tunable", False) and k.startswith(REC_PREFIX)
    }


    names: List[str] = []
    lows: List[float] = []
    highs: List[float] = []

    for feat_name, meta in tunable.items():
        cur_val = None if current is None else current.get(feat_name)
        lo, hi = _local_bounds(meta, cur_val, step_pct)
        names.append(feat_name)
        lows.append(lo)
        highs.append(hi)


    base_mid = {k: v.get("mid") for k, v in fb.items()}


    for ek in ENV_LOCKED:
        if fixed_context and ek in fixed_context:
            base_mid[ek] = fixed_context[ek]

    def objective(trial: optuna.trial.Trial) -> float:

        cand: Dict[str, float] = {}
        for i, feat in enumerate(names):
            cand[feat] = trial.suggest_float(feat, lows[i], highs[i])

        merged: Dict[str, Any] = dict(base_mid)
        merged.update(cand)
        if fixed_context:
            merged.update(fixed_context)


        fe_vals = _derive_engineered_features(merged)
        merged.update(fe_vals)

        yhat = model_service.predict(type_code, merged)

        score = 0.0
        for p in POINTS:
            yh = yhat.get(p)
            if yh is None or (isinstance(yh, float) and (math.isnan(yh) or math.isinf(yh))):

                score += 5_000.0
                continue

            lo = float(specs[p]["lo"])
            hi = float(specs[p]["hi"])
            tgt = float(specs[p]["target"])
            w = float(specs[p]["w"])
            span = max(1.0, hi - lo)


            score += w * ((yh - tgt) / span) ** 2


            if yh < lo:
                score += 10.0 * (lo - yh) / span
            elif yh > hi:
                score += 10.0 * (yh - hi) / span

        return float(score)

    return objective


# ----------------------------------------------------------
# Public API: Optimize tunable machine parameters
# ----------------------------------------------------------
def optimize(model_service,
             type_code: str,
             specs: dict,
             fixed_context: dict,
             n_trials: int = 150,
             timeout: int | None = 5,
             current: dict | None = None,
             step_pct: float = 0.02):
    """
    Runs an Optuna TPE optimization to find the best machine parameter settings
    that achieve desired thickness targets.

    - model_service: final trained model (must expose .predict(type_code, features))
    - type_code: product type
    - specs: dict for A..G with lo/hi/target/w
    - fixed_context: environment & other locked inputs (Temperature/Humidity/Valve are respected)
    - current: current machine settings (for local windowing)
    - step_pct: local window size as % of (hi-lo) around current value
    """

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    objective = make_objective(model_service, type_code, specs, fixed_context, current, step_pct)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params if study.best_trial is not None else {}

    fb = get_feature_bounds(type_code) or {}


    merged = {k: v.get("mid") for k, v in fb.items()}
    merged.update(best_params)
    if fixed_context:
        merged.update(fixed_context)
    merged.update(_derive_engineered_features(merged))


    predicted = model_service.predict(type_code, merged)

    best_score = float(study.best_value) if study.best_trial is not None else float("inf")
    trials = len(study.trials)

    return best_params, predicted, best_score, trials, None
