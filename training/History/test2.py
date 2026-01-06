# run in the same environment before starting training
import numpy as np, pandas as pd, os
from collections import Counter
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Tuple, Optional
from utils import (
    read_table, add_engineered_features, build_feature_lists, select_features,
    build_preprocessor, build_cv_splits, THICKNESS_COLS, winsorize_targets,
    _collect_specs_from_df, save_artifacts as save_artifacts_utils,
    MODEL_DIR as UTILS_MODEL_DIR, MODEL_VERSION as UTILS_MODEL_VERSION
)

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
            X[c] = X[c].astype(str).fillna("__NA__")
    return X

DATA_PATH = os.getenv("TRAIN_CSV", "./data/data.csv")

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

X_raw = df[feat_cols].copy()
X_clean = sanitize_feature_dataframe(X_raw, numeric_cols, categorical_cols)
pre = build_preprocessor(numeric_cols, categorical_cols)
print("Fitting preprocessor...")
pre.fit(X_clean)
X_all = pre.transform(X_clean)
if hasattr(X_all, "shape"):
    print("Preprocessor produced:", X_all.shape)
else:
    print("Preprocessor produced object type:", type(X_all))

Y = df[THICKNESS_COLS].ffill().fillna(0.0).values.astype(float)

def diagnose_df(df, feat_cols, label_cols, max_show=12):
    X = df[feat_cols].copy()
    Y = df[label_cols].copy()

    print("ROWS:", len(df))
    # - dtype summary
    print("\nFeature dtype counts:")
    print(X.dtypes.value_counts())

    # - For each column, compute NaNs, unique counts, and whether it is numeric-coercible
    stats = []
    for c in X.columns:
        ser = X[c]
        n_nan = ser.isna().sum()
        n_unique = ser.nunique(dropna=False)
        # try to coerce to numeric safely
        coerced = pd.to_numeric(ser, errors="coerce")
        n_coerce_na = coerced.isna().sum()
        is_numeric_like = (n_coerce_na < len(ser)) and (ser.dtype.kind in ('i','u','f') or n_coerce_na < len(ser))
        stats.append((c, str(ser.dtype), n_nan, n_unique, int(is_numeric_like), int(np.isinf(coerced.fillna(0)).any())))

    stats_df = pd.DataFrame(stats, columns=["col","dtype","n_nan","n_unique","numeric_like","had_inf"]).set_index("col")
    # show problematic cols: many NaNs, tiny n_unique, mixed types
    print("\nColumns with many NaNs (top):")
    print(stats_df.sort_values("n_nan", ascending=False).head(max_show))

    print("\nColumns with tiny number of uniques (<=2):")
    print(stats_df[stats_df["n_unique"] <= 2].sort_values("n_unique").head(max_show))

    print("\nColumns that look numeric-like but had Infs after coercion:")
    print(stats_df[stats_df["had_inf"] == 1].head(max_show))

    # Mixed Python types sampling for quick inspection
    mixed = {}
    for c in X.columns:
        s = X[c].dropna().head(50).map(lambda v: type(v).__name__).unique()
        if len(s) > 1:
            mixed[c] = list(s)
    print("\nColumns with mixed python types (sample):", list(mixed.keys())[:max_show])

    # Labels
    print("\nLabel stats:")
    for c in Y.columns:
        vals = Y[c].dropna()
        print(f"{c}: n={len(vals)}, nan={Y[c].isna().sum()}, min={vals.min() if len(vals) else None}, max={vals.max() if len(vals) else None}, std={vals.std() if len(vals) else None}, uniq={vals.nunique() if len(vals) else 0}")

    return stats_df, mixed

# Example usage:
# stats_df, mixed = safe_diagnose_df(df, feat_cols, THICKNESS_COLS)


# Example call:
# diagnose_df(df, feat_cols, THICKNESS_COLS)

import lightgbm as lgb
from lightgbm import LGBMRegressor
sample_n = min(2000, X_all.shape[0])
Xs = X_all[:sample_n]
ys = Y[:sample_n, 0]   # test point A

dtrain = lgb.Dataset(Xs, label=ys)
from lightgbm import LGBMRegressor

lgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1
}

try:
    model = LGBMRegressor(**lgb_params)
    # sklearn wrapper accepts early_stopping_rounds in many builds:
    model.fit(Xs, ys, eval_set=[(Xs, ys)], eval_metric="rmse", early_stopping_rounds=100, verbose=50)
    # if early_stopping_rounds triggers TypeError for your build, call without it:
except TypeError:
    model = LGBMRegressor(**lgb_params)
    model.fit(Xs, ys)
# predictions:
preds = model.predict(Xs)

print("Small LGB test done.")
