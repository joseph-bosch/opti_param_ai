# train_auto_opt.py (updated: robust handling for XGBoost/LightGBM fit signature differences)
"""
Automatic hyperparameter optimization using Optuna (robust preprocessing).
This version detects whether the installed XGBoost/LightGBM versions support
early stopping via sklearn .fit(...) or via callbacks and falls back safely.

Usage:
    python train_auto_opt.py
"""

import os
import sys
import inspect
import optuna
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv()

# Project utils - keep the same API as your repo's utils.py
from utils import (
    read_table, add_engineered_features, build_feature_lists, select_features,
    build_preprocessor, build_cv_splits, THICKNESS_COLS, winsorize_targets,
    _collect_specs_from_df, MODEL_DIR, MODEL_VERSION
)

# ML libs
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from scipy import sparse as sp_sparse

# Config (env override)
DATA_PATH = os.getenv("TRAIN_CSV", "./data/data.csv")
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50000"))
CV_S = int(os.getenv("N_SPLITS", "3"))
CV_METHOD = os.getenv("CV_METHOD", "timeseries")
GAP = int(os.getenv("GAP", "4"))
TARGET_CLIP_Q = tuple(float(x.strip()) for x in os.getenv("TARGET_CLIP_Q", "0.02,0.98").split(","))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
EARLY_STOP = int(os.getenv("EARLY_STOP_ROUNDS", "100"))

# Keep track of printed warnings to avoid spamming logs
_printed_warnings = set()

def _warn_once(key: str, msg: str):
    if key not in _printed_warnings:
        print(msg, file=sys.stderr)
        _printed_warnings.add(key)

# ----------------- Helpers -----------------

def detect_mixed_types(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict]:
    mixed = {}
    for c in cols:
        if c not in df.columns:
            continue
        vals = df[c].dropna()
        if vals.empty:
            continue
        types = vals.map(lambda x: type(x)).unique().tolist()
        if len(types) > 1:
            samples = vals.head(10).tolist()
            mixed[c] = {"types": [t.__name__ for t in types], "samples": samples}
    return mixed

def sanitize_feature_dataframe(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    mixed_before = detect_mixed_types(X, [c for c in categorical_cols if c in X.columns])
    if mixed_before:
        print("Detected mixed types in categorical columns (before coercion):")
        for col, info in mixed_before.items():
            print(f" - {col}: types={info['types']}, samples={info['samples'][:6]}")
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("__NA__")
    return X

def safe_build_cv_splits(df: pd.DataFrame, n_splits=3, cv_method="timeseries", gap=0):
    try:
        return build_cv_splits(df, n_splits=n_splits, cv_method=cv_method, gap=gap)
    except TypeError:
        try:
            return build_cv_splits(df)
        except Exception as e:
            raise RuntimeError(f"Unable to call build_cv_splits: {e}")

def wnrmse_from_preds(df: pd.DataFrame, preds: Dict[str, np.ndarray], specs: Dict) -> Tuple[float, Dict[str, float]]:
    per_point = {}
    total_w = 0.0
    total = 0.0
    labels = list("ABCDEFG")
    for p_label, tcol in zip(labels, THICKNESS_COLS):
        arr = preds.get(tcol)
        if arr is None:
            per_point[p_label] = float("nan")
            continue
        mask = ~np.isnan(arr)
        if mask.sum() == 0:
            per_point[p_label] = float("nan")
            continue
        y = df.loc[mask, tcol].values
        yh = arr[mask]
        rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
        sp = specs.get(p_label, {})
        lo = float(sp.get("lower_um", np.nanmin(y)))
        hi = float(sp.get("upper_um", np.nanmax(y)))
        span = max(1.0, hi - lo)
        w = float(sp.get("weight", 1.0))
        nrmse = rmse / span
        per_point[p_label] = nrmse
        total += w * nrmse
        total_w += w
    return total / max(1e-9, total_w), per_point


# ----------------- Load + preprocess -----------------

print("Loading data:", DATA_PATH)
df = read_table(DATA_PATH)
df = df.dropna(subset=THICKNESS_COLS, how="all")

if "Type" not in df.columns:
    df["Type"] = df.index.astype(str)
else:
    df["Type"] = df["Type"].astype(str)

# Winsorize targets if requested
ql, qh = TARGET_CLIP_Q
if (ql, qh) != (0.0, 1.0):
    print(f"Winsorizing targets at quantiles: {ql}, {qh}")
    df = winsorize_targets(df, ql, qh)

# Engineered features
df, eng_cols = add_engineered_features(df)

# Build features lists
base_feat_cols, base_numeric_cols, base_categorical_cols = build_feature_lists(df)
feat_cols, numeric_cols, categorical_cols, always_keep = select_features(df, base_numeric_cols, base_categorical_cols)

print(f"Initial features: {len(base_feat_cols)} (numeric={len(base_numeric_cols)}, categorical={len(base_categorical_cols)})")
print(f"Selected features: {len(feat_cols)} (numeric={len(numeric_cols)}, categorical={len(categorical_cols)})")
print(f"Always-kept ({len(always_keep)}): {always_keep}")

# Sanitize and preprocess
X_raw = df[feat_cols].copy()
X_clean = sanitize_feature_dataframe(X_raw, numeric_cols, categorical_cols)
pre = build_preprocessor(numeric_cols, categorical_cols)
print("Fitting preprocessor to sanitized feature table...")
pre.fit(X_clean)
X_all = pre.transform(X_clean)
if sp_sparse.issparse(X_all):
    print("Preprocessor produced sparse matrix of shape", X_all.shape)
else:
    print("Preprocessor produced array of shape", getattr(X_all, "shape", None))

Y = df[THICKNESS_COLS].ffill().fillna(0.0).values.astype(float)
specs = _collect_specs_from_df(df)
splits = safe_build_cv_splits(df, n_splits=CV_S, cv_method=CV_METHOD, gap=GAP)

# ----------------- Robust model factories -----------------

def _supports_fit_arg(estimator_or_class, arg_name: str) -> bool:
    try:
        sig = inspect.signature(estimator_or_class.fit)
        return arg_name in sig.parameters
    except Exception:
        return False

def xgb_model_factory(params, Xtr, ytr_j, Xva, yva_j):
    """
    Try sklearn XGBRegressor.fit(... early_stopping_rounds ...)
    If the local XGBoost wrapper does not accept early_stopping_rounds, use xgb.train with DMatrix and callbacks.
    """
    try:
        # prefer sklearn wrapper if it accepts early_stopping_rounds
        if _supports_fit_arg(XGBRegressor, "early_stopping_rounds"):
            model = XGBRegressor(**params)
            model.fit(Xtr, ytr_j, eval_set=[(Xva, yva_j)], early_stopping_rounds=EARLY_STOP, verbose=False)
            return model.predict(Xva)
        # if fit accepts 'callbacks' we can use xgb.callback.EarlyStopping
        elif _supports_fit_arg(XGBRegressor, "callbacks"):
            model = XGBRegressor(**params)
            es = xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True, metric_name="rmse", data_name="validation_1")
            model.fit(Xtr, ytr_j, eval_set=[(Xtr, ytr_j), (Xva, yva_j)], callbacks=[es], verbose=False)
            return model.predict(Xva)
        else:
            # Fallback: use the low-level xgb.train with DMatrix + callback (works across versions)
            dtrain = xgb.DMatrix(Xtr, label=ytr_j)
            dval = xgb.DMatrix(Xva, label=yva_j)
            # prepare xgboost-native params: copy but remove sklearn-only keys
            xgb_params = params.copy()
            for k in ("n_jobs", "nthread", "random_state"):
                xgb_params.pop(k, None)
            num_boost_round = xgb_params.pop("n_estimators", 1000)
            # ensure objective present
            xgb_params.setdefault("objective", "reg:squarederror")
            evallist = [(dtrain, "train"), (dval, "validation")]
            bst = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round,
                            evals=evallist, callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOP)])
            return bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
    except Exception as e:
        # bubble up so Optuna can catch and penalize trial
        raise RuntimeError(f"XGB factory failed: {e}")

def lgb_model_factory(params, Xtr, ytr_j, Xva, yva_j):
    """
    Try LGBMRegressor.fit(... early_stopping_rounds ...)
    If that fails, try lgb.train with callbacks; else fallback to fit without early stopping.
    """
    try:
        # prefer sklearn wrapper
        if _supports_fit_arg(LGBMRegressor, "early_stopping_rounds"):
            model = LGBMRegressor(**params)
            model.fit(Xtr, ytr_j, eval_set=[(Xva, yva_j)], eval_metric="rmse",
                      early_stopping_rounds=EARLY_STOP, verbose=False)
            return model.predict(Xva)
        # some builds accept 'callbacks'
        elif _supports_fit_arg(LGBMRegressor, "callbacks"):
            model = LGBMRegressor(**params)
            # Use lightgbm callback for early stopping
            es_cb = lgb.callback.early_stopping(stopping_rounds=EARLY_STOP, verbose=False)
            model.fit(Xtr, ytr_j, eval_set=[(Xva, yva_j)], eval_metric="rmse",
                      callbacks=[es_cb], verbose=False)
            return model.predict(Xva)
        else:
            # Try low-level lgb.train (may or may not support early_stopping_rounds depending on build)
            dtrain = lgb.Dataset(Xtr, label=ytr_j)
            dval = lgb.Dataset(Xva, label=yva_j, reference=dtrain)
            low_params = params.copy()
            num_boost_round = low_params.pop("n_estimators", 1000)
            # remove sklearn-only keys
            for k in ("verbosity", "random_state", "n_jobs", "nthread"):
                low_params.pop(k, None)
            low_params.setdefault("objective", "regression")
            try:
                bst = lgb.train(low_params, dtrain, num_boost_round=num_boost_round,
                                valid_sets=[dtrain, dval],
                                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
                return bst.predict(Xva, num_iteration=bst.best_iteration)
            except TypeError as te:
                # low-level train failed to accept early_stopping_rounds; fall back to plain LGBMRegressor fit without ES
                _warn_once("lgb_no_es", f"[LGB fallback] early stopping not supported in this LightGBM build: {te}; training without ES.")
                model = LGBMRegressor(**params)
                model.fit(Xtr, ytr_j)
                return model.predict(Xva)
    except Exception as e:
        raise RuntimeError(f"LGB factory failed: {e}")

# ----------------- CV wrapper -----------------

def _preds_for_model_across_cv_for_params(model_factory, params: dict):
    preds = {tc: np.full(len(df), np.nan, dtype=float) for tc in THICKNESS_COLS}
    for tr_idx, va_idx, info in splits:
        tr_mask = np.isin(df.index.values, tr_idx)
        va_mask = np.isin(df.index.values, va_idx)
        if va_mask.sum() == 0:
            continue
        Xtr = X_all[tr_mask]
        Xva = X_all[va_mask]
        ytr = Y[tr_mask]
        yva = Y[va_mask]
        for j, tc in enumerate(THICKNESS_COLS):
            preds[tc][va_mask] = model_factory(params, Xtr, ytr[:, j], Xva, yva[:, j])
    return preds

# ----------------- Optuna objectives -----------------

def xgb_objective(trial):
    try:
        params = {
            "n_estimators": trial.suggest_int("x_n_estimators", 500, 4000),
            "learning_rate": trial.suggest_float("x_learning_rate", 1e-4, 0.2, log=True),
            "max_depth": trial.suggest_int("x_max_depth", 3, 10),
            "subsample": trial.suggest_float("x_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("x_colsample", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("x_reg_lambda", 1e-6, 10.0, log=True),
            "reg_alpha": trial.suggest_float("x_reg_alpha", 1e-6, 10.0, log=True),
            "min_child_weight": trial.suggest_int("x_min_child_weight", 1, 30),
            "gamma": trial.suggest_float("x_gamma", 0.0, 2.0),
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
        }
        preds = _preds_for_model_across_cv_for_params(xgb_model_factory, params)
        wnrmse, per_point = wnrmse_from_preds(df, preds, specs)
        trial.set_user_attr("wnrmse", float(wnrmse))
        trial.set_user_attr("per_point", per_point)
        return float(wnrmse)
    except Exception as e:
        print(f"[xgb_objective exception] {e}", file=sys.stderr)
        return float(1e6)

def lgb_objective(trial):
    try:
        params = {
            "n_estimators": trial.suggest_int("l_n_estimators", 500, 4000),
            "learning_rate": trial.suggest_float("l_learning_rate", 1e-4, 0.2, log=True),
            "num_leaves": trial.suggest_int("l_num_leaves", 16, 256),
            "max_depth": trial.suggest_int("l_max_depth", -1, 12),
            "subsample": trial.suggest_float("l_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("l_colsample", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("l_reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("l_reg_lambda", 1e-6, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        }
        preds = _preds_for_model_across_cv_for_params(lgb_model_factory, params)
        wnrmse, per_point = wnrmse_from_preds(df, preds, specs)
        trial.set_user_attr("wnrmse", float(wnrmse))
        trial.set_user_attr("per_point", per_point)
        return float(wnrmse)
    except Exception as e:
        print(f"[lgb_objective exception] {e}", file=sys.stderr)
        return float(1e6)

def combined_objective(trial):
    try:
        algo = trial.suggest_categorical("algo", ["xgb", "lgb"])
        if algo == "xgb":
            return xgb_objective(trial)
        else:
            return lgb_objective(trial)
    except Exception as e:
        print(f"[combined_objective exception] {e}", file=sys.stderr)
        return float(1e6)

# ----------------- Run Optuna -----------------

def run_optuna(n_trials=N_TRIALS):
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    print(f"Starting Optuna study with n_trials={n_trials} ...")
    study.optimize(combined_objective, n_trials=n_trials)
    print("Optuna finished. Best trial:")
    print(study.best_trial.params)
    print("Best value (wnrmse):", study.best_value)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(study, os.path.join(MODEL_DIR, f"optuna_study_{MODEL_VERSION}.joblib"))
    return study

if __name__ == "__main__":
    study = run_optuna(N_TRIALS)
