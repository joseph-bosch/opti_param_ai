# utils.py
# Shared utilities: IO, feature engineering, CV, metrics, artifact saving.
# Adapted from your original train_xgb.py (keeps same behavior and column names).

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold, GroupKFold, train_test_split
from sqlalchemy import create_engine, text

# ----- Constants (same as your original)
THICKNESS_COLS = [
    "Thickness Point A","Thickness Point B","Thickness Point C",
    "Thickness Point D","Thickness Point E","Thickness Point F","Thickness Point G"
]
ID_COLS = ["Batch_ID"]
TIME_COL = "Date_Produced"
TYPE_COL = "Type"

# ----- Environment-driven defaults (override with os.environ if desired)
import os
MODEL_DIR = os.getenv("MODEL_DIR", "../models")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.2.0")
TARGET_CLIP_Q = tuple(float(x.strip()) for x in os.getenv("TARGET_CLIP_Q", "0.02,0.98").split(","))
USE_FEAT_ENG = int(os.getenv("USE_FEAT_ENG", "1"))
USE_TIME_INDEX = int(os.getenv("USE_TIME_INDEX", "1"))
TIME_INDEX_SCALE = float(os.getenv("TIME_INDEX_SCALE", "1.0"))
USE_EXPOSURES = int(os.getenv("USE_EXPOSURES", "0"))
EXPO_SPEED_FLOOR_Q = float(os.getenv("EXPO_SPEED_FLOOR_Q", "0.20"))
EXPO_CLIP_QHI = float(os.getenv("EXPO_CLIP_QHI", "0.995"))
EXPO_USE_LOG = int(os.getenv("EXPO_USE_LOG", "1"))
EXPOSURE_EPS = float(os.getenv("EXPOSURE_EPS", "1e-3"))
DROP_CONSTANTS = int(os.getenv("DROP_CONSTANTS", "1"))
CORR_DROP_THR = float(os.getenv("CORR_DROP_THR", "0.995"))
SELECT_TOP_NUMERIC = int(os.getenv("SELECT_TOP_NUMERIC", "999"))
ALWAYS_KEEP_PATTERNS = [s.strip() for s in os.getenv(
    "ALWAYS_KEEP_PATTERNS",
    "Axis-X_Upper_Point,Axis-Z_Upper_Point,Axis-Z_Lower_Point,"
    "Gun1_Voltage,Gun2_Voltage,Gun1_Powder,Gun2_Powder,Time_Index"
).split(",") if s.strip()]
USE_MONO = int(os.getenv("USE_MONO", "0"))
MONO_POS = [s.strip() for s in os.getenv("MONO_POS", "Voltage,Powder,total_powder,total_voltage,exposure").split(",") if s.strip()]
MONO_NEG = [s.strip() for s in os.getenv("MONO_NEG", "Speed").split(",") if s.strip()]
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# MSSQL defaults (can be overridden)
MSSQL_DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
MSSQL_USER = os.getenv("MSSQL_USER", "")
MSSQL_PASS = os.getenv("MSSQL_PASSWORD", "")
MSSQL_HOST = os.getenv("MSSQL_HOST", "(localdb)\\MSSQLLocalDB")
MSSQL_DB = os.getenv("MSSQL_DB", "OptiParamAI")


# ---------- IO ----------
def make_engine():
    drv = MSSQL_DRIVER.replace(" ", "+")
    url = f"mssql+pyodbc://{MSSQL_USER}:{MSSQL_PASS}@{MSSQL_HOST}/{MSSQL_DB}?driver={drv}&Encrypt=no&TrustServerCertificate=yes"
    return create_engine(url, pool_pre_ping=True, fast_executemany=True)


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    if ext in [".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"]:
        if ext == ".xls":
            df = pd.read_excel(path)
        else:
            df = pd.read_excel(path, engine="openpyxl")
    elif ext in [".csv", ".txt"]:
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            df = pd.read_csv(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    return df

# ---------- Feature lists and selection ----------
def build_feature_lists(df: pd.DataFrame):
    exclude = set(ID_COLS + [TIME_COL] + THICKNESS_COLS)
    feat_cols = [c for c in df.columns if c not in exclude]
    if TYPE_COL in feat_cols:
        feat_cols = [TYPE_COL] + [c for c in feat_cols if c != TYPE_COL]
    numeric_cols = df[feat_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feat_cols if c not in numeric_cols]
    return feat_cols, numeric_cols, categorical_cols

def _matches_any(name: str, patterns: List[str]) -> bool:
    lname = name.lower()
    return any(p.lower() in lname for p in patterns)

def _drop_constants(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = []
    for c in cols:
        s = df[c].dropna()
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep

def _drop_high_corr(df: pd.DataFrame, cols: List[str], thr: float, keep_cols: List[str]) -> List[str]:
    if not cols or thr >= 1.0:
        return cols
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for column in upper.columns:
        if any(upper[column] > thr) and (column not in keep_cols):
            to_drop.add(column)
    return [c for c in cols if c not in to_drop]

def _rank_numeric_by_spearman(df: pd.DataFrame, numeric_cols: List[str], top_n: int) -> List[str]:
    if not numeric_cols:
        return []
    scores = {}
    for col in numeric_cols:
        vals = df[col]
        if vals.isna().all() or vals.nunique(dropna=True) <= 1:
            scores[col] = 0.0; continue
        rs = []
        for t in THICKNESS_COLS:
            r, _ = spearmanr(df[col], df[t], nan_policy="omit")
            rs.append(abs(r) if not np.isnan(r) else 0.0)
        scores[col] = float(np.mean(rs))
    ranked = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return ranked[:max(1, min(top_n, len(ranked)))]

def select_features(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    always_keep = [c for c in numeric_cols if _matches_any(c, ALWAYS_KEEP_PATTERNS)]
    pruned_num = numeric_cols
    if DROP_CONSTANTS:
        before = len(pruned_num)
        pruned_num = _drop_constants(df, pruned_num)
        print(f"Dropped {before - len(pruned_num)} constant numeric features.")
    pruned_num = _drop_high_corr(df, pruned_num, thr=CORR_DROP_THR, keep_cols=always_keep)
    top_num = _rank_numeric_by_spearman(df, pruned_num, top_n=max(10, SELECT_TOP_NUMERIC))
    ordered = []
    for c in always_keep + top_num:
        if c not in ordered:
            ordered.append(c)
    feat_cols = ordered + categorical_cols
    return feat_cols, ordered, categorical_cols, always_keep

# ---------- Engineered features ----------
def add_engineered_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    eng_cols = []
    if not USE_FEAT_ENG:
        return df, eng_cols

    letters = list("ABCDEF")
    total_pow_all = None
    total_volt_all = None
    exposure_sum = None

    for L in letters:
        g1p = f"Rec_{L}_Gun1_Powder"
        g2p = f"Rec_{L}_Gun2_Powder"
        g1v = f"Rec_{L}_Gun1_Voltage"
        g2v = f"Rec_{L}_Gun2_Voltage"
        spd = f"Rec_{L}_Axis-Z_Speed"

        if (g1p in df.columns) or (g2p in df.columns):
            tpow = df.get(g1p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) + \
                   df.get(g2p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
            col = f"FE_{L}_Total_Powder"
            df[col] = tpow
            eng_cols.append(col)
            total_pow_all = (tpow if total_pow_all is None else total_pow_all + tpow)

        if (g1v in df.columns) or (g2v in df.columns):
            tvol = df.get(g1v, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) + \
                   df.get(g2v, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
            col = f"FE_{L}_Total_Voltage"
            df[col] = tvol
            eng_cols.append(col)
            total_volt_all = (tvol if total_volt_all is None else total_volt_all + tvol)

        if USE_EXPOSURES and (spd in df.columns) and ((g1p in df.columns) or (g2p in df.columns)):
            sp = df[spd].astype(float)
            floor = pd.to_numeric(sp.quantile(EXPO_SPEED_FLOOR_Q), errors="coerce")
            if pd.isna(floor) or floor <= 0:
                floor = EXPOSURE_EPS
            sp_safe = np.maximum(sp.fillna(floor), floor)

            tpow = df.get(f"FE_{L}_Total_Powder", None)
            if tpow is None:
                tpow = df.get(g1p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) + \
                       df.get(g2p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)

            expo = tpow / sp_safe
            try:
                hi = float(np.nanquantile(expo.astype(float), EXPO_CLIP_QHI))
            except Exception:
                hi = np.nanmax(expo.astype(float))
            if not np.isfinite(hi) or hi <= 0:
                hi = expo.astype(float).fillna(0).quantile(0.99)
            expo = np.clip(expo.astype(float), 0, hi)

            if EXPO_USE_LOG:
                expo = np.log1p(expo)
                col = f"FE_{L}_Exposure_Log"
            else:
                col = f"FE_{L}_Exposure"

            df[col] = expo
            eng_cols.append(col)
            exposure_sum = (expo if exposure_sum is None else exposure_sum + expo)

    if total_pow_all is not None:
        df["FE_All_Total_Powder"] = total_pow_all; eng_cols.append("FE_All_Total_Powder")
    if total_volt_all is not None:
        df["FE_All_Total_Voltage"] = total_volt_all; eng_cols.append("FE_All_Total_Voltage")
    if USE_EXPOSURES and (exposure_sum is not None):
        df["FE_All_Exposure_Index"] = exposure_sum; eng_cols.append("FE_All_Exposure_Index")

    if USE_TIME_INDEX and (TIME_COL in df.columns) and pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        t0 = df[TIME_COL].min()
        days = (df[TIME_COL] - t0).dt.total_seconds() / 86400.0
        df["FE_Time_Index"] = days * TIME_INDEX_SCALE
        eng_cols.append("FE_Time_Index")

    return df, eng_cols

# ---------- Preprocessor ----------
def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                               ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop")
    return pre

def _get_tx_feature_names(pre: ColumnTransformer) -> List[str]:
    names = []
    try:
        names.extend(list(pre.transformers_[0][2]))
    except Exception:
        pass
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        names.extend(list(ohe.get_feature_names_out()))
    except Exception:
        pass
    return names

# ---------- CV splitting ----------
def build_ts_splits_with_gap(df: pd.DataFrame, n_splits=3, gap=0) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    if TIME_COL not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        raise ValueError("Time-series CV requires a datetime Date_Produced column.")
    ordered = df.sort_values(TIME_COL)
    order_idx = ordered.index.values
    n = len(order_idx)
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    rel = np.arange(n)
    for tr_rel, va_rel in tss.split(rel):
        if gap > 0 and len(va_rel) > 0:
            cut = max(0, va_rel[0] - gap)
            tr_rel = tr_rel[tr_rel < cut]
        tr_idx = order_idx[tr_rel]
        va_idx = order_idx[va_rel]
        overlap = set(tr_idx).intersection(set(va_idx))
        assert len(overlap) == 0, f"Train/valid overlap detected: {len(overlap)}"
        info = dict(
            train_rows=int(len(tr_idx)), valid_rows=int(len(va_idx)),
            train_start=str(ordered.loc[tr_idx, TIME_COL].min()) if len(tr_idx) else None,
            train_end=str(ordered.loc[tr_idx, TIME_COL].max()) if len(tr_idx) else None,
            valid_start=str(ordered.loc[va_idx, TIME_COL].min()) if len(va_idx) else None,
            valid_end=str(ordered.loc[va_idx, TIME_COL].max()) if len(va_idx) else None,
            gap=int(gap)
        )
        splits.append((tr_idx, va_idx, info))
    return splits

def build_cv_splits(df: pd.DataFrame, n_splits=3, cv_method="timeseries", gap=0, group_col="Batch_ID") -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    n = len(df)
    if n < 10 or n_splits < 2:
        return [(np.arange(n), np.arange(0), dict(train_rows=n, valid_rows=0))]
    if cv_method == "timeseries" and TIME_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        return build_ts_splits_with_gap(df, n_splits=n_splits, gap=gap)
    if cv_method == "group" and group_col in df.columns:
        groups = df[group_col].astype(str).fillna("NA")
        gkf = GroupKFold(n_splits=n_splits)
        return [(tr, va, dict(train_rows=len(tr), valid_rows=len(va))) for tr, va in gkf.split(df, groups=groups)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return [(tr, va, dict(train_rows=len(tr), valid_rows=len(va))) for tr, va in kf.split(df)]

# ---------- Specs / metrics ----------
def _collect_specs_from_df(df: pd.DataFrame) -> Dict[str,Dict]:
    d = {}
    for p_label, tcol in zip(list("ABCDEFG"), THICKNESS_COLS):
        vals = df[tcol].dropna()
        if len(vals) == 0: continue
        lo, med, hi = float(vals.quantile(0.1)), float(vals.median()), float(vals.quantile(0.9))
        d[p_label] = {"lower_um": lo, "upper_um": hi, "default_target_um": med, "weight": 1.0}
    for k in ("B","C"):
        if k in d: d[k]["weight"] = 1.2
    return d

def _compute_weighted_nrmse(valid_df: pd.DataFrame, yhat: Dict[str,np.ndarray], specs: Dict[str,Dict]) -> Tuple[float, Dict[str,float]]:
    per_point = {}
    total_w = 0.0
    total = 0.0
    for p_label, tcol in zip(list("ABCDEFG"), THICKNESS_COLS):
        y = valid_df[tcol].values
        yh = yhat[tcol]
        rmse = float(np.sqrt(np.mean((yh - y)**2)))
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

# ---------- Persistence helpers ----------
def save_artifacts(pre, models: Dict[str, object], feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str], model_dir: str = MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pre, os.path.join(model_dir, "preprocessor.joblib"))
    # models keyed by human tcol (e.g. "Thickness Point A")
    for tcol, model in models.items():
        fname = f"xgb_{tcol.split()[-1]}.joblib"
        joblib.dump(model, os.path.join(model_dir, fname))
    info = {"feature_cols": feature_cols, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "model_version": MODEL_VERSION}
    with open(os.path.join(model_dir, "feature_info.json"), "w") as f:
        json.dump(info, f, indent=2)

def save_model_object(obj, name: str, model_dir: str = MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(obj, os.path.join(model_dir, name))

# ---------- Small helper ----------
def winsorize_targets(df: pd.DataFrame, qlow: float, qhi: float) -> pd.DataFrame:
    if not (0.0 <= qlow < qhi <= 1.0): return df
    if (qlow, qhi) == (0.0, 1.0): return df
    df2 = df.copy()
    for t in THICKNESS_COLS:
        if t in df2.columns:
            lo = df2[t].quantile(qlow); hi = df2[t].quantile(qhi)
            df2[t] = df2[t].clip(lower=lo, upper=hi)
    return df2
