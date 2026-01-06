#!/usr/bin/env python3
# train_auto_opt_full_fixed.py
"""
Comprehensive Optuna-based hyperparameter optimization pipeline (per-algorithm studies)

New capability:
 - MULTI_ALGO_MODE=roundrobin → rotate algos in batches (e.g., 5k trials each)
   until every algo reaches its target number of trials.
 - Control order with ALGO_ORDER (default: nn,xgb,cat,lgb)
 - Control batch size with ROUND_TRIALS (default: 5000)
 - Control targets with OPTUNA_TRIALS (global default) or per-algo:
       TARGET_TRIALS_NN / _XGB / _CAT / _LGB

Existing:
 - Separate Optuna studies per algo
 - Safe checkpointing, per-algo CSV logs, stacking final models
 - Parallel & sequential modes still supported (unchanged)
"""

import os
import sys
import json
import time
import inspect
import joblib
import traceback
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ML libs
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool as CatPool
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# sklearn
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.base import clone

# DB (optional)
from sqlalchemy import create_engine, text

# Project utils
from utils import (
    read_table, add_engineered_features, build_feature_lists, select_features,
    build_preprocessor, build_cv_splits, THICKNESS_COLS, winsorize_targets,
    _collect_specs_from_df, save_artifacts as save_artifacts_utils,
    MODEL_DIR as UTILS_MODEL_DIR, MODEL_VERSION as UTILS_MODEL_VERSION
)

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
# -------------------- CONFIG / ENV --------------------
MODEL_DIR = os.getenv("MODEL_DIR", UTILS_MODEL_DIR or "./models")
MODEL_VERSION = os.getenv("MODEL_VERSION", UTILS_MODEL_VERSION or "v1.0.0")
DATA_PATH = BASE_DIR / "Data_1.xlsx"

OPTUNA_STORAGE_TYPE = os.getenv("OPTUNA_STORAGE_TYPE", "mssql").lower()
OPTUNA_SQLITE_PATH = os.getenv("OPTUNA_SQLITE_PATH", os.path.join(MODEL_DIR, "optuna_study.db"))

OPTUNA_STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"opti_{MODEL_VERSION}")
# Acts as the DEFAULT target trials per algo (unless per-algo overrides are set below)
# OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20000"))
OPTUNA_PARALLEL_JOBS = int(os.getenv("OPTUNA_PARALLEL_JOBS", "1"))

# Scheduling:
#   "parallel"  → old behavior (per-algo studies in parallel)
#   "sequential"→ old behavior (per-algo studies one-by-one FULL trials)
#   "roundrobin"→ NEW behavior (batched rotations until per-algo targets reached)
MULTI_ALGO_MODE = os.getenv("MULTI_ALGO_MODE", "roundrobin").lower()
ALGO_ORDER = [a.strip() for a in os.getenv("ALGO_ORDER", "nn,xgb,cat,lgb").split(",") if a.strip()]
ROUND_TRIALS = int(os.getenv("ROUND_TRIALS", "5000"))  # per pass, per algo

# Optional per-algo targets (fallback to OPTUNA_TRIALS)
TARGET_TRIALS_NN  = int(os.getenv("TARGET_TRIALS_NN"))
TARGET_TRIALS_XGB = int(os.getenv("TARGET_TRIALS_XGB"))
TARGET_TRIALS_CAT = int(os.getenv("TARGET_TRIALS_CAT"))
TARGET_TRIALS_LGB = int(os.getenv("TARGET_TRIALS_LGB"))

SAVE_TRIALS_TO_DB = int(os.getenv("SAVE_TRIALS_TO_DB", "0"))
TRIALS_TABLE = os.getenv("TRIALS_TABLE", "optuna_trials")
BACKUP_EVERY_N_TRIALS = int(os.getenv("BACKUP_EVERY_N_TRIALS", "15000"))

MSSQL_HOST = os.getenv("MSSQL_HOST", "")
MSSQL_DB = os.getenv("MSSQL_DB", "")
MSSQL_USER = os.getenv("MSSQL_USER", "")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "")
MSSQL_DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")

EARLY_STOP = int(os.getenv("EARLY_STOP_ROUNDS", "110"))

CHECKPOINT_AFTER_TRIALS = int(os.getenv("CHECKPOINT_AFTER_TRIALS", "10000"))
CHECKPOINT_EVERY_N_TRIALS = int(os.getenv("CHECKPOINT_EVERY_N_TRIALS", "10000"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "45"))
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

META_KERAS_UNITS = int(os.getenv("META_KERAS_UNITS", "64"))
META_KERAS_LAYERS = int(os.getenv("META_KERAS_LAYERS", "2"))
META_KERAS_DROPOUT = float(os.getenv("META_KERAS_DROPOUT", "0.1"))
META_KERAS_EPOCHS = int(os.getenv("META_KERAS_EPOCHS", "200"))
META_KERAS_BATCH = int(os.getenv("META_KERAS_BATCH", "64"))

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# -------------------- DB URL --------------------
def build_mssql_sqlalchemy_url() -> str:
    from urllib.parse import quote_plus
    driver_q = quote_plus(MSSQL_DRIVER)
    user = MSSQL_USER or ""
    pwd = MSSQL_PASSWORD or ""
    host = MSSQL_HOST or "localhost"
    db = MSSQL_DB or ""
    if user and pwd:
        return f"mssql+pyodbc://{quote_plus(user)}:{quote_plus(pwd)}@{host}/{db}?driver={driver_q}&Encrypt=no&TrustServerCertificate=yes"
    else:
        return f"mssql+pyodbc://{host}/{db}?driver={driver_q}"

if OPTUNA_STORAGE_TYPE == "mssql":
    OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", build_mssql_sqlalchemy_url())
else:
    OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", f"sqlite:///{OPTUNA_SQLITE_PATH}")

print("Using Optuna storage:", OPTUNA_STORAGE)

# -------------------- HELPERS (types / preprocessing) --------------------
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
            mixed[c] = {"types": [t.__name__ for t in types], "samples": vals.head(10).tolist()}
    return mixed

def sanitize_feature_dataframe(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    mixed_before = detect_mixed_types(X, [c for c in categorical_cols if c in X.columns])
    if mixed_before:
        print("[Sanitize] Mixed types in categorical columns:", mixed_before)
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].fillna("__NA__").astype("category")
    return X

def to_numpy_dense(X):
    from scipy import sparse as sp_sparse
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X.values
    if sp_sparse.issparse(X):
        return X.toarray()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def _encode_cats_for_xgb(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        return X
    X_enc = X.copy()
    cat_cols = [c for c in X_enc.columns if isinstance(X_enc[c].dtype, pd.CategoricalDtype)]
    for c in cat_cols:
        codes = X_enc[c].cat.codes
        codes = codes.replace({-1: np.nan})
        X_enc[c] = codes
    return X_enc

# -------------------- LOAD + PREPROCESS --------------------
print("Loading data:", DATA_PATH)
df = read_table(DATA_PATH)
df = df.dropna(subset=THICKNESS_COLS, how="all")
if "Type" not in df.columns:
    df["Type"] = df.index.astype(str)
else:
    df["Type"] = df["Type"].astype(str)

ql, qh = tuple(float(x.strip()) for x in os.getenv("TARGET_CLIP_Q", "0.02,0.98").split(","))
if (ql, qh) != (0.0, 1.0):
    print(f"Winsorizing targets at {ql},{qh}")
    df = winsorize_targets(df, ql, qh)

df, eng_cols = add_engineered_features(df)
base_feat_cols, base_numeric_cols, base_categorical_cols = build_feature_lists(df)
feat_cols, numeric_cols, categorical_cols, always_keep = select_features(df, base_numeric_cols, base_categorical_cols)

print(f"Initial features: {len(base_feat_cols)} (numeric={len(base_numeric_cols)}, categorical={len(base_categorical_cols)})")
print(f"Selected features: {len(feat_cols)} (numeric={len(numeric_cols)}, categorical={len(categorical_cols)})")
print("Always keep features count:", len(always_keep))
print("Categorical columns:", categorical_cols)

X_raw = df[feat_cols].copy()
X_clean = sanitize_feature_dataframe(X_raw, numeric_cols, categorical_cols)

expected_engineered = [
    "FE_All_Total_Voltage","FE_D_Total_Voltage","FE_All_Total_Powder","FE_C_Total_Voltage","FE_Time_Index",
]
for col in expected_engineered:
    if col not in X_clean.columns:
        X_clean[col] = 0.0

X_clean = X_clean.reindex(sorted(X_clean.columns), axis=1)

fill_defaults = {}
for c in X_clean.columns:
    if pd.api.types.is_numeric_dtype(X_clean[c]):
        fill_defaults[c] = X_clean[c].median(skipna=True)
    elif isinstance(X_clean[c].dtype, pd.CategoricalDtype) or X_clean[c].dtype == object:
        fill_defaults[c] = "__NA__"
    else:
        fill_defaults[c] = 0.0

for col in X_clean.select_dtypes(include=["category"]).columns:
    if "__NA__" not in X_clean[col].cat.categories:
        X_clean[col] = X_clean[col].cat.add_categories(["__NA__"])

for col, val in fill_defaults.items():
    if X_clean[col].isna().all():
        X_clean[col] = val
X_clean = X_clean.fillna(fill_defaults)
X_clean = X_clean.reindex(sorted(X_clean.columns), axis=1)

pre = build_preprocessor(numeric_cols, categorical_cols)
print("Fitting preprocessor...")
pre.fit(X_clean)

fill_defaults = {}
for c in X_clean.columns:
    if pd.api.types.is_numeric_dtype(X_clean[c]):
        fill_defaults[c] = float(X_clean[c].median(skipna=True))
    else:
        fill_defaults[c] = "__NA__"
fill_defaults_path = os.path.join(MODEL_DIR, "fill_defaults.json")
with open(fill_defaults_path, "w") as fh:
    json.dump(fill_defaults, fh, indent=2)
print(f"[FINAL] Saved fill_defaults → {fill_defaults_path}")

X_all = pre.transform(X_clean)               # for NN/linear
X_tree_all = X_clean[feat_cols].copy()       # for trees (with categories)

if hasattr(X_all, "shape"):
    print("Preprocessor produced:", X_all.shape)
else:
    print("Preprocessor produced object type:", type(X_all))

Y = df[THICKNESS_COLS].ffill().fillna(0.0).values.astype(float)
specs = _collect_specs_from_df(df)
splits = build_cv_splits(df)

# -------------------- Model factories --------------------
def _supports_fit_arg(estimator_or_class, arg_name: str) -> bool:
    try:
        sig = inspect.signature(estimator_or_class.fit)
        return arg_name in sig.parameters
    except Exception:
        return False

def _categorical_feature_indices_from_df(df_columns: List[str], categorical_cols_list: List[str]) -> List[int]:
    idxs = []
    for c in categorical_cols_list:
        if c in df_columns:
            idxs.append(df_columns.index(c))
    return idxs

def xgb_model_factory(params, Xtr, ytr_j, Xva, yva_j):
    try:
        Xtr_enc = _encode_cats_for_xgb(Xtr) if isinstance(Xtr, pd.DataFrame) else Xtr
        Xva_enc = _encode_cats_for_xgb(Xva) if isinstance(Xva, pd.DataFrame) else Xva

        if _supports_fit_arg(XGBRegressor, "early_stopping_rounds"):
            model = XGBRegressor(**params)
            model.fit(Xtr_enc, ytr_j, eval_set=[(Xva_enc, yva_j)], early_stopping_rounds=EARLY_STOP, verbose=False)
            return model.predict(Xva_enc)

        elif _supports_fit_arg(XGBRegressor, "callbacks"):
            model = XGBRegressor(**params)
            es = xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True, metric_name="rmse", data_name="validation_1")
            model.fit(Xtr_enc, ytr_j, eval_set=[(Xtr_enc, ytr_j), (Xva_enc, yva_j)], callbacks=[es], verbose=False)
            return model.predict(Xva_enc)

        else:
            dtrain = xgb.DMatrix(Xtr_enc, label=ytr_j)
            dval = xgb.DMatrix(Xva_enc, label=yva_j)
            xgb_params = params.copy()
            for k in ("n_jobs", "nthread", "random_state"):
                xgb_params.pop(k, None)
            num_boost_round = xgb_params.pop("n_estimators", 1000)
            xgb_params.setdefault("objective", "reg:squarederror")
            bst = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round,
                            evals=[(dtrain, "train"), (dval, "validation")],
                            callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOP)])
            return bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
    except Exception as e:
        raise RuntimeError(f"XGB factory failed: {e}")

def lgb_model_factory(params, Xtr, ytr_j, Xva, yva_j):
    try:
        local_params = params.copy() if params is not None else {}
        local_params.setdefault("verbosity", -1)

        cat_feat = None
        if isinstance(Xtr, pd.DataFrame):
            cat_feat = [c for c in categorical_cols if c in Xtr.columns]

        if _supports_fit_arg(LGBMRegressor, "early_stopping_rounds"):
            model = LGBMRegressor(**local_params)
            fit_kwargs = {"eval_set": [(Xva, yva_j)], "eval_metric": "rmse", "early_stopping_rounds": EARLY_STOP}
            if cat_feat:
                fit_kwargs["categorical_feature"] = cat_feat
            model.fit(Xtr, ytr_j, **fit_kwargs)
            return model.predict(Xva)

        elif _supports_fit_arg(LGBMRegressor, "callbacks"):
            model = LGBMRegressor(**local_params)
            es_cb = lgb.callback.early_stopping(stopping_rounds=EARLY_STOP)
            if cat_feat:
                model.fit(Xtr, ytr_j, eval_set=[(Xva, yva_j)], eval_metric="rmse", callbacks=[es_cb], categorical_feature=cat_feat)
            else:
                model.fit(Xtr, ytr_j, eval_set=[(Xva, yva_j)], eval_metric="rmse", callbacks=[es_cb])
            return model.predict(Xva)

        else:
            dtrain = lgb.Dataset(Xtr, label=ytr_j, categorical_feature=cat_feat if cat_feat else None)
            dval = lgb.Dataset(Xva, label=yva_j, reference=dtrain, categorical_feature=cat_feat if cat_feat else None)
            low = local_params.copy()
            num_boost = low.pop("n_estimators", 1000)
            for k in ("verbosity", "random_state", "n_jobs", "nthread"):
                low.pop(k, None)
            low.setdefault("objective", "regression")
            bst = lgb.train(low, dtrain, num_boost_round=num_boost, valid_sets=[dtrain, dval],
                            callbacks=[lgb.callback.early_stopping(stopping_rounds=EARLY_STOP)])
            return bst.predict(Xva, num_iteration=bst.best_iteration)
    except Exception as e:
        raise RuntimeError(f"LGB factory failed: {e}")

def cat_model_factory(params, Xtr, ytr_j, Xva, yva_j):
    try:
        cat_feat = [c for c in categorical_cols if c in Xtr.columns]
        if not cat_feat:
            print("[CatBoost] Warning: No categorical features detected.")
        else:
            print(f"[CatBoost] Using categorical features: {cat_feat}")

        model = CatBoostRegressor(**params)
        try:
            train_pool = CatPool(Xtr, ytr_j, cat_features=cat_feat if cat_feat else None)
            eval_pool = CatPool(Xva, yva_j, cat_features=cat_feat if cat_feat else None)
            model.fit(train_pool, eval_set=eval_pool,
                      use_best_model=True, early_stopping_rounds=EARLY_STOP, verbose=False)
        except TypeError:
            model.fit(Xtr, ytr_j, eval_set=(Xva, yva_j),
                      cat_features=cat_feat, use_best_model=True,
                      early_stopping_rounds=EARLY_STOP, verbose=False)
        except Exception:
            model.fit(Xtr, ytr_j, cat_features=cat_feat, verbose=False)
        return model.predict(Xva)
    except Exception as e:
        raise RuntimeError(f"CatBoost factory failed: {e}")

def make_keras_model(input_dim: int, hp: dict):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(int(input_dim),)))
    for _ in range(int(hp.get("n_layers", 2))):
        model.add(
            layers.Dense(int(hp.get("units", 128)),
                         activation=hp.get("activation", "relu"),
                         kernel_regularizer=keras.regularizers.l2(hp.get("weight_decay", 0.0)))
        )
        if float(hp.get("dropout", 0.0)) > 0.0:
            model.add(layers.Dropout(float(hp.get("dropout", 0.0))))
    model.add(layers.Dense(1, activation="linear"))
    opt = keras.optimizers.Adam(learning_rate=float(hp.get("lr", 1e-3)))
    model.compile(loss="mse", optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])
    return model

def nn_model_factory(hp, Xtr, ytr_j, Xva, yva_j):
    def to_dense_np(X):
        if hasattr(X, "toarray"):
            return X.toarray()
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values.astype(float)
        return np.asarray(X, dtype=float)

    Xtr = to_dense_np(Xtr)
    Xva = to_dense_np(Xva)
    ytr_j = np.asarray(ytr_j, dtype=float).reshape(-1)
    yva_j = np.asarray(yva_j, dtype=float).reshape(-1)

    input_dim = int(Xtr.shape[1])
    model = make_keras_model(input_dim, hp)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=max(5, int(EARLY_STOP // 10)), restore_best_weights=True, verbose=0)
    ]
    epochs = int(hp.get("epochs", META_KERAS_EPOCHS))
    batch = int(hp.get("batch_size", META_KERAS_BATCH))
    model.fit(Xtr, ytr_j, validation_data=(Xva, yva_j), epochs=epochs, batch_size=batch, callbacks=callbacks, verbose=0)
    preds = model.predict(Xva).reshape(-1)
    return preds, model

# -------------------- CV / metrics --------------------
def _preds_for_model_across_cv_for_params(model_factory, params: dict, use_tree_df: bool=False, nn_special=False):
    preds = {tc: np.full(len(df), np.nan, dtype=float) for tc in THICKNESS_COLS}
    models_for_debug = {}

    for tr_idx, va_idx, info in splits:
        tr_mask = np.isin(df.index.values, tr_idx)
        va_mask = np.isin(df.index.values, va_idx)
        if va_mask.sum() == 0:
            continue

        if use_tree_df:
            Xtr = X_tree_all.iloc[tr_mask]
            Xva = X_tree_all.iloc[va_mask]
        else:
            Xtr = X_all[tr_mask]
            Xva = X_all[va_mask]

        ytr = Y[tr_mask]
        yva = Y[va_mask]

        for j, tc in enumerate(THICKNESS_COLS):
            if nn_special:
                preds_j, model_obj = model_factory(params, Xtr, ytr[:, j], Xva, yva[:, j])
                preds[tc][va_mask] = preds_j
                models_for_debug[tc] = model_obj
            else:
                preds[tc][va_mask] = model_factory(params, Xtr, ytr[:, j], Xva, yva[:, j])

    return (preds, models_for_debug) if nn_special else preds

def wnrmse_from_preds(df_local: pd.DataFrame, preds: Dict[str, np.ndarray], specs_local: Dict) -> Tuple[float, Dict[str,float]]:
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
        y = df_local.loc[mask, tcol].values
        yh = arr[mask]
        rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
        sp = specs_local.get(p_label, {})
        lo = float(sp.get("lower_um", np.nanmin(y)))
        hi = float(sp.get("upper_um", np.nanmax(y)))
        span = max(1.0, hi - lo)
        w = float(sp.get("weight", 1.0))
        nrmse = rmse / span
        per_point[p_label] = nrmse
        total += w * nrmse
        total_w += w
    return total / max(1e-9, total_w), per_point

# -------------------- Optuna objectives --------------------
def xgb_objective(trial):
    try:
        params = {
            "n_estimators": trial.suggest_int("x_n_estimators", 500, 3000),
            "learning_rate": trial.suggest_float("x_learning_rate", 1e-4, 0.2, log=True),
            "max_depth": trial.suggest_int("x_max_depth", 3, 9),
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
        preds = _preds_for_model_across_cv_for_params(xgb_model_factory, params, use_tree_df=True)
        wnrmse, per_point = wnrmse_from_preds(df, preds, specs)
        trial.set_user_attr("wnrmse", float(wnrmse))
        trial.set_user_attr("per_point", per_point)
        return float(wnrmse)
    except Exception as e:
        print("[xgb_objective exception]", e, file=sys.stderr)
        return float(1e6)

def lgb_objective(trial):
    try:
        params = {
            "n_estimators": trial.suggest_int("l_n_estimators", 500, 3000),
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
        preds = _preds_for_model_across_cv_for_params(lgb_model_factory, params, use_tree_df=True)
        wnrmse, per_point = wnrmse_from_preds(df, preds, specs)
        trial.set_user_attr("wnrmse", float(wnrmse))
        trial.set_user_attr("per_point", per_point)
        return float(wnrmse)
    except Exception as e:
        print("[lgb_objective exception]", e, file=sys.stderr)
        return float(1e6)

def cat_objective(trial):
    try:
        params = {
            "iterations": trial.suggest_int("c_iterations", 500, 3000),
            "depth": trial.suggest_int("c_depth", 4, 10),
            "learning_rate": trial.suggest_float("c_lr", 1e-4, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("c_l2", 1e-6, 10.0, log=True),
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False
        }
        preds = _preds_for_model_across_cv_for_params(
            lambda p, Xtr, ytr, Xva, yva: cat_model_factory(params, Xtr, ytr, Xva, yva),
            params={}, use_tree_df=True
        )
        wnrmse, per_point = wnrmse_from_preds(df, preds, specs)
        trial.set_user_attr("wnrmse", float(wnrmse))
        trial.set_user_attr("per_point", per_point)
        return float(wnrmse)
    except Exception as e:
        print("[cat_objective exception]", e, file=sys.stderr)
        return float(1e6)

def nn_objective(trial):
    try:
        hp = {
            "n_layers": trial.suggest_int("nn_n_layers", 1, 4),
            "units": trial.suggest_categorical("nn_units", [32, 64, 128, 256]),
            "dropout": trial.suggest_float("nn_dropout", 0.0, 0.5),
            "lr": trial.suggest_float("nn_lr", 1e-5, 1e-2, log=True),
            "epochs": trial.suggest_int("nn_epochs", 50, 400),
            "batch_size": trial.suggest_categorical("nn_batch", [32, 64, 128]),
            "activation": trial.suggest_categorical("nn_act", ["relu", "elu", "selu"])
        }
        preds = _preds_for_model_across_cv_for_params(
            lambda p, Xtr, ytr, Xva, yva: nn_model_factory(p, Xtr, ytr, Xva, yva)[0],
            hp, use_tree_df=False, nn_special=False
        )
        wnrmse, per_point = wnrmse_from_preds(df, preds, specs)
        trial.set_user_attr("wnrmse", float(wnrmse))
        trial.set_user_attr("per_point", per_point)
        return float(wnrmse)
    except Exception as e:
        print("[nn_objective exception]", e, file=sys.stderr)
        return float(1e6)

# -------------------- Per-algo helpers --------------------
ALGOS = [a.strip() for a in os.getenv("ALGOS", "xgb,lgb,cat,nn").split(",") if a.strip()]
OBJECTIVES = {"xgb": xgb_objective, "lgb": lgb_objective, "cat": cat_objective, "nn": nn_objective}

def _study_name_for(algo: str) -> str:
    return f"{OPTUNA_STUDY_NAME}_{algo}"

def _final_joblib_for(algo: str) -> str:
    return os.path.join(MODEL_DIR, f"optuna_study_final_{MODEL_VERSION}_{algo}.joblib")

def _io_paths_for_algo(algo: str):
    trials_csv = os.path.join(MODEL_DIR, f"optuna_trials_{MODEL_VERSION}_{algo}.csv")
    dash_csv   = os.path.join(MODEL_DIR, f"optuna_dashboard_{MODEL_VERSION}_{algo}.csv")
    return trials_csv, dash_csv

# -------------------- Persist / logging --------------------
import csv
_db_engine = None

if SAVE_TRIALS_TO_DB:
    try:
        print("[DB] Creating engine for trial logging.")
        _db_engine = create_engine(build_mssql_sqlalchemy_url(), fast_executemany=True)
    except Exception as e:
        print("[DB] Failed to create engine for trial logging:", e, file=sys.stderr)
        _db_engine = None

def append_trial_to_csv(frozen_trial, trials_csv_path: str):
    write_header = not os.path.exists(trials_csv_path)
    row = {
        "trial_number": frozen_trial.number,
        "state": frozen_trial.state.name,
        "value": float(frozen_trial.value) if frozen_trial.value is not None else None,
        "params": json.dumps(frozen_trial.params),
        "datetime_complete": pd.Timestamp.utcnow().isoformat()
    }
    with open(trials_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def update_dashboard_csv(study, dashboard_csv_path: str):
    try:
        df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
        total = len(df_trials)
        completed = (df_trials["state"] == "COMPLETE").sum() if "state" in df_trials.columns else 0
        best_val = float(study.best_value) if study.best_value is not None else None
        try:
            best_algo = study.study_name.rsplit("_", 1)[-1]
        except Exception:
            best_algo = None
        mean_val = float(df_trials["value"].dropna().astype(float).mean()) if "value" in df_trials.columns and not df_trials["value"].dropna().empty else None
        row = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "study": study.study_name,
            "total_trials": total,
            "completed_trials": int(completed),
            "best_value": best_val,
            "best_algo": best_algo,
            "mean_value": mean_val
        }
        write_header = not os.path.exists(dashboard_csv_path)
        with open(dashboard_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print("[dashboard] failed to update:", e, file=sys.stderr)

def insert_trial_into_db(frozen_trial, study_name: str):
    if not SAVE_TRIALS_TO_DB or _db_engine is None:
        return
    try:
        with _db_engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {TRIALS_TABLE} (study_name, trial_number, state, value, params, datetime_complete)
                VALUES(:s, :n, :st, :v, :p, :d)
            """), {
                "s": study_name,
                "n": int(frozen_trial.number),
                "st": frozen_trial.state.name,
                "v": float(frozen_trial.value) if frozen_trial.value is not None else None,
                "p": json.dumps(frozen_trial.params),
                "d": pd.Timestamp.utcnow().isoformat()
            })
    except Exception as e:
        print("[DB insert] failed:", e, file=sys.stderr)

def backup_study(study, dest_path):
    try:
        joblib.dump(study, dest_path)
        print("[Backup] saved study to", dest_path)
    except Exception as e:
        print("[Backup] failed:", e, file=sys.stderr)

_best_checkpoint_value_by_algo = defaultdict(lambda: float("inf"))

def _build_constructor_kwargs_for_algo(algo: str, params: dict) -> dict:
    out = {}
    if algo == "xgb":
        for k, v in params.items():
            if k.startswith("x_"): out[k[2:]] = v
        if "max_depth" in out:
            out["max_depth"] = int(out["max_depth"])
        return out
    elif algo == "lgb":
        for k, v in params.items():
            if k.startswith("l_"):
                out_key = "colsample_bytree" if k[2:] == "colsample" else k[2:]
                out[out_key] = v
        return out
    elif algo == "cat":
        mapping = {"c_iterations": "iterations", "c_depth": "depth", "c_lr": "learning_rate", "c_l2": "l2_leaf_reg"}
        for k, v in params.items():
            if k in mapping:
                out[mapping[k]] = v
            elif k.startswith("c_"):
                out[k[2:]] = v
        return out
    else:
        return {}

def train_and_save_models_from_trial(trial, algo_hint=None):
    print("[Checkpoint] Training final models for trial", getattr(trial, "number", "<dict>"))
    params = trial.params if hasattr(trial, "params") else dict(trial)
    algo = params.get("algo", algo_hint)
    X_full_tree = X_tree_all
    Y_full = Y

    artifacts_dir = Path(MODEL_DIR) / f"checkpoint_trial_{getattr(trial,'number',int(time.time()))}_{algo}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, artifacts_dir / "preprocessor.joblib")
    print("[Checkpoint] preprocessor saved.")
    models = {}

    for j, tcol in enumerate(THICKNESS_COLS):
        y_full_j = Y_full[:, j]
        try:
            if algo == "xgb":
                X_enc = _encode_cats_for_xgb(X_full_tree)
                model = XGBRegressor(
                    **_build_constructor_kwargs_for_algo("xgb", params),
                    objective="reg:squarederror", tree_method="hist",
                    random_state=RANDOM_STATE, n_jobs=-1
                )
                try:
                    cutoff = int(len(X_enc) * 0.9)
                    Xtr, Xva = X_enc.iloc[:cutoff], X_enc.iloc[cutoff:]
                    ytr, yva = y_full_j[:cutoff], y_full_j[cutoff:]
                    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=EARLY_STOP, verbose=False)
                except Exception:
                    model.fit(X_enc, y_full_j)
                models[tcol] = model

            elif algo == "lgb":
                lparams = _build_constructor_kwargs_for_algo("lgb", params)
                lparams.setdefault("verbosity", -1)
                model = LGBMRegressor(**lparams, random_state=RANDOM_STATE, n_jobs=-1)
                cat_feat = [c for c in categorical_cols if c in X_full_tree.columns]
                try:
                    cutoff = int(len(X_full_tree) * 0.9)
                    model.fit(
                        X_full_tree.iloc[:cutoff], y_full_j[:cutoff],
                        eval_set=[(X_full_tree.iloc[cutoff:], y_full_j[cutoff:])],
                        early_stopping_rounds=EARLY_STOP, categorical_feature=cat_feat
                    )
                except Exception:
                    model.fit(X_full_tree, y_full_j, categorical_feature=cat_feat)
                models[tcol] = model

            elif algo == "cat":
                cparams = _build_constructor_kwargs_for_algo("cat", params)
                cparams.setdefault("verbose", False)
                cparams.setdefault("allow_writing_files", False)
                cparams.setdefault("random_seed", RANDOM_STATE)
                model = CatBoostRegressor(**cparams)
                try:
                    cutoff = int(len(X_full_tree) * 0.9)
                    model.fit(
                        X_full_tree.iloc[:cutoff], y_full_j[:cutoff],
                        eval_set=(X_full_tree.iloc[cutoff:], y_full_j[cutoff:]),
                        use_best_model=True, early_stopping_rounds=EARLY_STOP, verbose=False
                    )
                except Exception:
                    model.fit(X_full_tree, y_full_j, verbose=False)
                models[tcol] = model

            elif algo == "nn":
                hp = {
                    "n_layers": int(params.get("nn_n_layers", 2)),
                    "units": int(params.get("nn_n_units", params.get("nn_units", 128))),
                    "dropout": float(params.get("nn_n_dropout", params.get("nn_dropout", 0.1))),
                    "lr": float(params.get("nn_lr", 1e-3)),
                    "epochs": int(params.get("nn_epochs", 100)),
                    "batch_size": int(params.get("nn_batch", params.get("nn_batch_size", 64))),
                    "activation": params.get("nn_act", "relu")
                }
                X_train = X_all
                if hasattr(X_train, "toarray"):
                    X_train = X_train.toarray()
                model = make_keras_model(X_train.shape[1], hp)
                cb = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, verbose=0)
                try:
                    model.fit(X_train, y_full_j, epochs=hp["epochs"], batch_size=hp["batch_size"], callbacks=[cb], verbose=0)
                except Exception as e:
                    print("[Checkpoint NN train] warning:", e)
                models[tcol] = model
            else:
                raise ValueError("Unknown algo for checkpointing: " + str(algo))

            out_path = artifacts_dir / f"{algo}_{tcol.replace(' ','_')}.joblib"
            try:
                joblib.dump(models[tcol], out_path)
            except Exception:
                if algo == "nn":
                    models[tcol].save(str(artifacts_dir / f"{algo}_{tcol.replace(' ','_')}.tf"))
                else:
                    raise
        except Exception as e:
            print(f"[Checkpoint] failed training model for {tcol}: {e}", file=sys.stderr)

    with open(artifacts_dir / "checkpoint_info.json", "w") as fh:
        json.dump({"trial": getattr(trial,'number', None), "algo": algo, "params": params}, fh, indent=2)
    print("[Checkpoint] Completed and saved to", artifacts_dir)
    return str(artifacts_dir)

def make_trial_logger_and_checkpoint(algo: str):
    trials_csv, dash_csv = _io_paths_for_algo(algo)
    study_name = _study_name_for(algo)

    def _cb(study, frozen_trial):
        try:
            append_trial_to_csv(frozen_trial, trials_csv)
        except Exception as e:
            print("[trial_logger] CSV append failed:", e, file=sys.stderr)
        try:
            insert_trial_into_db(frozen_trial, study_name)
        except Exception as e:
            print("[trial_logger] DB insert failed:", e, file=sys.stderr)
        try:
            update_dashboard_csv(study, dash_csv)
        except Exception as e:
            print("[trial_logger] dashboard update failed:", e, file=sys.stderr)
        try:
            if BACKUP_EVERY_N_TRIALS > 0 and (frozen_trial.number % BACKUP_EVERY_N_TRIALS == 0):
                backup_path = os.path.join(MODEL_DIR, f"optuna_study_backup_{MODEL_VERSION}_{algo}_trial{frozen_trial.number}.joblib")
                backup_study(study, backup_path)
        except Exception as e:
            print("[trial_logger] backup failed:", e, file=sys.stderr)

        # Safe checkpointing
        try:
            bv = study.best_value
            if (bv is not None) and np.isfinite(bv) and (bv < 1e5):
                if getattr(study.best_trial, "number", 0) >= CHECKPOINT_AFTER_TRIALS:
                    if bv < _best_checkpoint_value_by_algo[algo]:
                        _best_checkpoint_value_by_algo[algo] = float(bv)
                        print(f"[Checkpoint Callback][{algo}] New best wnrmse:", _best_checkpoint_value_by_algo[algo],
                              "trial:", study.best_trial.number)
                        try:
                            if CHECKPOINT_EVERY_N_TRIALS == 0 or (study.best_trial.number % CHECKPOINT_EVERY_N_TRIALS == 0):
                                train_and_save_models_from_trial(study.best_trial, algo_hint=algo)
                            else:
                                print(f"[Checkpoint][{algo}] Skipping heavy checkpoint until divisible by {CHECKPOINT_EVERY_N_TRIALS}")
                        except Exception as e:
                            print("[Checkpoint] failed to train/save models:", e, traceback.format_exc(), file=sys.stderr)
        except Exception as e:
            print("[trial_logger] checkpoint logic failed:", e, file=sys.stderr)

    return _cb

# -------------------- Optuna study runners --------------------
def create_or_resume_study_for_algo(algo: str):
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction="minimize",
                                study_name=_study_name_for(algo),
                                sampler=sampler,
                                pruner=pruner,
                                storage=OPTUNA_STORAGE,
                                load_if_exists=True)
    return study

def run_optuna_for_algo(algo: str, n_trials: int, callbacks=None, n_jobs: int = OPTUNA_PARALLEL_JOBS):
    if algo not in OBJECTIVES:
        raise ValueError(f"Unknown algo: {algo}")
    study = create_or_resume_study_for_algo(algo)
    print(f"[Optuna][{algo}] Starting batch: n_trials={n_trials}")
    if callbacks is None:
        callbacks = [make_trial_logger_and_checkpoint(algo)]
    study.optimize(OBJECTIVES[algo], n_trials=n_trials, callbacks=callbacks, n_jobs=n_jobs)
    save_path = _final_joblib_for(algo)
    joblib.dump(study, save_path)
    print(f"[Optuna][{algo}] Batch complete. best_trial={study.best_trial.number} best_value={study.best_value}")
    return study

def load_or_resume_final_study_for_algo(algo: str):
    path = _final_joblib_for(algo)
    if os.path.exists(path):
        return joblib.load(path)
    return optuna.load_study(study_name=_study_name_for(algo), storage=OPTUNA_STORAGE)

def collect_best_params_from_study(study) -> Optional[dict]:
    best = None
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        if best is None or (t.value is not None and t.value < best["value"]):
            best = {"value": float(t.value), "params": t.params}
    return None if best is None else best["params"]

# -------------------- FINAL TRAINING + STACKING --------------------
def train_final_models_and_stack(best_trials_info: dict):
    print("\n========================================================")
    print("[FINAL] Training final base models + stacking meta models")
    print("========================================================\n")

    pre_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    joblib.dump(pre, pre_path)
    print(f"[FINAL] Saved preprocessor → {pre_path}")

    # Save cat_mappings
    try:
        src_df = X_tree_all if isinstance(X_tree_all, pd.DataFrame) else (X_clean if isinstance(X_clean, pd.DataFrame) else (df if isinstance(df, pd.DataFrame) else None))
        cat_mappings = {}
        if src_df is None:
            print("[FINAL] cat_mappings: no source DataFrame found (skipping).")
        else:
            for col in (categorical_cols or []):
                if col not in src_df.columns:
                    continue
                try:
                    if isinstance(src_df[col].dtype, pd.CategoricalDtype):
                        cats = list(src_df[col].cat.categories)
                    else:
                        tmp = src_df[col].astype(str).fillna("__NA__")
                        cats = list(pd.Categorical(tmp).categories)
                    if cats:
                        cat_mappings[col] = cats
                except Exception as e:
                    print(f"[FINAL] cat_mappings: failed for {col}: {e}", file=sys.stderr)
        if cat_mappings:
            with open(os.path.join(MODEL_DIR, "cat_mappings.json"), "w") as fh:
                json.dump(cat_mappings, fh, indent=2)
            print(f"[FINAL] Saved cat_mappings")
    except Exception as e:
        print(f"[FINAL] cat_mappings generation failed: {e}", file=sys.stderr)

    stacking_preds = {tcol: pd.DataFrame(index=df.index) for tcol in THICKNESS_COLS}

    def get_factory_and_mode(algo, params):
        if algo == "xgb":
            p = _build_constructor_kwargs_for_algo("xgb", params)
            return (lambda Xtr, ytr, Xva, yva: xgb_model_factory(p, Xtr, ytr, Xva, yva), True)
        elif algo == "lgb":
            p = _build_constructor_kwargs_for_algo("lgb", params)
            return (lambda Xtr, ytr, Xva, yva: lgb_model_factory(p, Xtr, ytr, Xva, yva), True)
        elif algo == "cat":
            p = _build_constructor_kwargs_for_algo("cat", params)
            return (lambda Xtr, ytr, Xva, yva: cat_model_factory(p, Xtr, ytr, Xva, yva), True)
        elif algo == "nn":
            return (lambda Xtr, ytr, Xva, yva: nn_model_factory(params, Xtr, ytr, Xva, yva)[0], False)
        else:
            raise ValueError("Unknown algo: " + str(algo))

    for algo, params in best_trials_info.items():
        if not isinstance(params, dict) or len(params) == 0:
            print(f"[FINAL] Skipping {algo}: empty params.")
            continue

        print(f"\n[FINAL] Processing algorithm: {algo}")
        factory, use_tree = get_factory_and_mode(algo, params)

        print(f"[FINAL] Generating OOF stacking predictions for {algo} ...")
        for tr_idx, va_idx, _ in splits:
            tr_mask = np.isin(df.index.values, tr_idx)
            va_mask = np.isin(df.index.values, va_idx)
            if va_mask.sum() == 0:
                continue

            Xtr = (X_tree_all if use_tree else X_all)[tr_mask]
            Xva = (X_tree_all if use_tree else X_all)[va_mask]
            ytr = Y[tr_mask]
            yva = Y[va_mask]

            for j, tcol in enumerate(THICKNESS_COLS):
                try:
                    preds_va = factory(Xtr, ytr[:, j], Xva, yva[:, j])
                    stacking_preds[tcol].loc[df.index.values[va_mask], algo] = preds_va
                except Exception as e:
                    print(f"[FINAL] OOF failed {algo} {tcol}: {e}", file=sys.stderr)

        print(f"[FINAL] Training {algo} on 100% of dataset ...")
        for j, tcol in enumerate(THICKNESS_COLS):
            y_full_j = Y[:, j]
            try:
                model = None
                if algo == "xgb":
                    X_full_enc = _encode_cats_for_xgb(X_tree_all)
                    model = XGBRegressor(
                        **{k.replace("x_", ""): params[k] for k in params if k.startswith("x_")},
                        objective="reg:squarederror", tree_method="hist",
                        random_state=RANDOM_STATE, n_jobs=-1
                    )
                    model.fit(X_full_enc, y_full_j)
                elif algo == "lgb":
                    model_kwargs = _build_constructor_kwargs_for_algo("lgb", params)
                    model_kwargs.setdefault("verbosity", -1)
                    model = LGBMRegressor(**model_kwargs, random_state=RANDOM_STATE)
                    cat_feat = [c for c in categorical_cols if c in X_tree_all.columns]
                    model.fit(X_tree_all, y_full_j, categorical_feature=cat_feat)
                elif algo == "cat":
                    model_kwargs = _build_constructor_kwargs_for_algo("cat", params)
                    model_kwargs.setdefault("random_seed", RANDOM_STATE)
                    model_kwargs.setdefault("verbose", False)
                    model_kwargs.setdefault("allow_writing_files", False)
                    model = CatBoostRegressor(**model_kwargs)
                    for c in categorical_cols:
                        if c in X_tree_all.columns and not isinstance(X_tree_all[c].dtype, pd.CategoricalDtype):
                            X_tree_all[c] = X_tree_all[c].astype("category")
                    cat_feat = [c for c in categorical_cols if c in X_tree_all.columns]
                    try:
                        train_pool = CatPool(X_tree_all, y_full_j, cat_features=cat_feat if cat_feat else None)
                        model.fit(train_pool, verbose=False)
                    except Exception as e:
                        print(f"[FINAL][CatBoost] Pool fit failed ({e}), retrying without Pool...")
                        model.fit(X_tree_all, y_full_j, cat_features=cat_feat, verbose=False)
                elif algo == "nn":
                    hp = {
                        "n_layers": int(params.get("nn_n_layers", 2)),
                        "units": int(params.get("nn_units", 128)),
                        "dropout": float(params.get("nn_dropout", 0.1)),
                        "lr": float(params.get("nn_lr", 1e-3)),
                        "epochs": int(params.get("nn_epochs", 100)),
                        "batch_size": int(params.get("nn_batch", 64)),
                        "activation": params.get("nn_act", "relu"),
                    }
                    Xf = X_all.toarray() if hasattr(X_all, "toarray") else X_all
                    model = make_keras_model(Xf.shape[1], hp)
                    model.fit(Xf, y_full_j, epochs=hp["epochs"], batch_size=hp["batch_size"], verbose=0)

                out_path = os.path.join(MODEL_DIR, f"{algo}_{tcol.replace(' ', '_')}.joblib")
                try:
                    joblib.dump(model, out_path)
                except Exception:
                    try:
                        model.save(out_path.replace(".joblib", ".tf"))
                    except Exception as ee:
                        print(f"[FINAL] Model save failed for {out_path}: {ee}", file=sys.stderr)
                print(f"[FINAL] Saved model → {out_path}")
            except Exception as e:
                print(f"[FINAL] FAILED training {algo} {tcol}: {e}", file=sys.stderr)

    print("\n[FINAL] Preparing stacking features...")
    meta_models_info = {}
    meta_selection = {}

    for tcol, df_preds in stacking_preds.items():
        for col in list(df_preds.columns):
            try:
                df_preds[col] = df_preds[col].fillna(df_preds[col].median())
            except Exception:
                df_preds[col] = df_preds[col].fillna(0.0)

        Xmeta = df_preds.values
        ymeta = df[tcol].values
        if Xmeta.shape[1] == 0:
            print(f"[META] No stacking features for {tcol} — skipped.")
            continue

        candidates = {
            "ridge": Ridge(alpha=1.0),
            "elasticnetcv": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=RANDOM_STATE, max_iter=5000)
        }

        best_name = None
        best_score = float("inf")
        results = {}
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        for name, model in candidates.items():
            scores = []
            for tr_i, va_i in kf.split(Xmeta):
                Xt, Xv = Xmeta[tr_i], Xmeta[va_i]
                yt, yv = ymeta[tr_i], ymeta[va_i]
                model_fit = clone(model)
                model_fit.fit(Xt, yt)
                pv = model_fit.predict(Xv)
                rmse = np.sqrt(np.mean((pv - yv) ** 2))
                scores.append(rmse)
            results[name] = float(np.mean(scores))
            if results[name] < best_score:
                best_score = results[name]
                best_name = name

        print(f"[META] {tcol}: selected {best_name} (cv_rmse={best_score:.6f})")
        meta_model = candidates[best_name]
        meta_model.fit(Xmeta, ymeta)
        meta_path = os.path.join(MODEL_DIR, f"stack_meta_{tcol.replace(' ', '_')}.joblib")
        joblib.dump(meta_model, meta_path)
        meta_models_info[tcol] = {"type": "sklearn", "name": best_name}
        meta_selection[tcol] = results

    stack_info = {
        "algos": list(best_trials_info.keys()),
        "features": feat_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "meta_selection": meta_selection
    }
    stack_info_path = os.path.join(MODEL_DIR, "stack_info.json")
    with open(stack_info_path, "w") as fh:
        json.dump(stack_info, fh, indent=2)

    print(f"\n✅ [FINAL] All stacking artifacts saved → {MODEL_DIR}")
    print("✅ [FINAL] Meta-learners saved")
    print("✅ [FINAL] Training complete!\n")

# -------------------- Drivers --------------------
def run_all_algos_and_train_sequential(algo_list: List[str], trials_each: int):
    studies = {}
    for algo in algo_list:
        try:
            studies[algo] = run_optuna_for_algo(algo, trials_each, None, OPTUNA_PARALLEL_JOBS)
        except Exception as e:
            print(f"[Driver][{algo}] Study failed:", e, file=sys.stderr)

    best_map = {}
    for algo in algo_list:
        st = studies.get(algo) or load_or_resume_final_study_for_algo(algo)
        params = collect_best_params_from_study(st)
        if params:
            params = dict(params); params["algo"] = algo
            best_map[algo] = params

    if not best_map:
        print("[Driver] No best params found — nothing to train.")
        return
    train_final_models_and_stack(best_map)

def run_all_algos_and_train_parallel(algo_list: List[str], trials_each: int, procs: int):
    studies = {}
    with ProcessPoolExecutor(max_workers=min(procs, len(algo_list))) as ex:
        futs = {ex.submit(run_optuna_for_algo, algo, trials_each, None, OPTUNA_PARALLEL_JOBS): algo for algo in algo_list}
        for fut in as_completed(futs):
            algo = futs[fut]
            try:
                studies[algo] = fut.result()
            except Exception as e:
                print(f"[Driver][{algo}] Study failed:", e, file=sys.stderr)

    best_map = {}
    for algo in algo_list:
        st = studies.get(algo) or load_or_resume_final_study_for_algo(algo)
        params = collect_best_params_from_study(st)
        if params:
            params = dict(params); params["algo"] = algo
            best_map[algo] = params

    if not best_map:
        print("[Driver] No best params found — nothing to train.")
        return
    train_final_models_and_stack(best_map)

# === NEW: Round-robin batched runner =========================================
from optuna.trial import TrialState

def _completed_trials_count(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

def _target_for_algo(algo: str) -> int:
    if algo == "nn":  return TARGET_TRIALS_NN
    if algo == "xgb": return TARGET_TRIALS_XGB
    if algo == "cat": return TARGET_TRIALS_CAT
    if algo == "lgb": return TARGET_TRIALS_LGB
    return OPTUNA_TRIALS

def run_roundrobin_batched(algo_order: List[str], batch_trials: int):
    # Initialize / resume studies
    studies: Dict[str, optuna.Study] = {}
    targets: Dict[str, int] = {a: _target_for_algo(a) for a in algo_order}

    for a in algo_order:
        if a not in OBJECTIVES:
            print(f"[RoundRobin] Skipping unknown algo: {a}")
            continue
        studies[a] = create_or_resume_study_for_algo(a)

    def _progress_str():
        parts = []
        for a in algo_order:
            st = studies.get(a)
            if st is None: continue
            done = _completed_trials_count(st)
            tgt = targets[a]
            parts.append(f"{a.upper()} {done}/{tgt}")
        return " | ".join(parts)

    print(f"[RoundRobin] Order: {', '.join(algo_order)} | Batch={batch_trials} | Targets={targets}")
    print("[RoundRobin] Progress:", _progress_str())

    # Loop until all algos reach their target
    while True:
        all_met = True
        for a in algo_order:
            st = studies.get(a)
            if st is None: 
                continue
            done = _completed_trials_count(st)
            tgt = targets[a]
            remaining = max(0, tgt - done)
            if remaining > 0:
                all_met = False
                n_this_batch = min(batch_trials, remaining)
                print(f"\n[RoundRobin] >>> {a.upper()}: running {n_this_batch} trials (progress {done}/{tgt})")
                studies[a] = run_optuna_for_algo(a, n_this_batch, None, OPTUNA_PARALLEL_JOBS)
                print("[RoundRobin] Progress:", _progress_str())
        if all_met:
            print("\n[RoundRobin] All algos reached targets.")
            break

    # Collect winners and train + stack
    best_map = {}
    for a in algo_order:
        st = studies.get(a) or load_or_resume_final_study_for_algo(a)
        params = collect_best_params_from_study(st)
        if params:
            params = dict(params); params["algo"] = a
            best_map[a] = params

    if not best_map:
        print("[RoundRobin] No best params found — nothing to train.")
        return
    train_final_models_and_stack(best_map)

# -------------------- Main --------------------
if __name__ == "__main__":
    FINAL_ONLY = int(os.getenv("FINAL_ONLY", "0"))

    if FINAL_ONLY == 1:
        print("[Main] FINAL_ONLY=1 → Loading final studies per algo...")
        best_map = {}
        # Use ALGO_ORDER to define which algos to include
        for algo in ALGO_ORDER:
            try:
                study = load_or_resume_final_study_for_algo(algo)
                params = collect_best_params_from_study(study)
                if params:
                    params = dict(params); params["algo"] = algo
                    best_map[algo] = params
                    try:
                        joblib.dump(study, _final_joblib_for(algo))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[Main] Could not load study for {algo}:", e, file=sys.stderr)

        if not best_map:
            raise RuntimeError("FINAL_ONLY=1 but no per-algo studies were found/resolved.")
        print("[Summary] Best per algo:", {k: v.get("algo", k) for k, v in best_map.items()})
        train_final_models_and_stack(best_map)
        sys.exit(0)

    # Scheduler selection
    if MULTI_ALGO_MODE == "roundrobin":
        run_roundrobin_batched(ALGO_ORDER, ROUND_TRIALS)
    elif MULTI_ALGO_MODE == "parallel":
        procs = int(os.getenv("ALGO_PROCESSES", str(min(4, len(ALGO_ORDER)))))
        run_all_algos_and_train_parallel(ALGO_ORDER, OPTUNA_TRIALS, procs)
    else:  # "sequential" (legacy: each algo gets full trials in sequence)
        run_all_algos_and_train_sequential(ALGO_ORDER, OPTUNA_TRIALS)






# -------------------- Entry point --------------------
# if __name__ == "__main__":
#     study = run_optuna(OPTUNA_TRIALS)
#     trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
#     best_per_algo = {}
#     for t in study.trials:
#         if t.state.name != "COMPLETE":
#             continue
#         algo = t.params.get("algo")
#         if algo not in best_per_algo or t.value < best_per_algo[algo]["value"]:
#             best_per_algo[algo] = {"value": t.value, "params": t.params}
#     print("[Summary] Best per algo:", {k:v["value"] for k,v in best_per_algo.items()})
#     if best_per_algo:
#         best_map = {algo: info["params"] for algo, info in best_per_algo.items()}
#         train_final_models_and_stack(best_map)
#     else:
#         print("[Main] No completed trials found to train final models.")

