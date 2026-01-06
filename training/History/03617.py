# if True:
#         # -*- coding: utf-8 -*-
#     """
#     Training script v1.2.4
#     - Non-overlapping time-series CV with gap + per-fold diagnostics and baseline
#     - Robust objective option (Pseudo-Huber) with automatic fallback
#     - Optional TimeIndex feature
#     - Optional domain-engineered features (totals and exposure proxies)
#     - Monotonic constraints (Voltage/Powder/Total/Exposure positive; Speed negative)
#     - Final XGBoost models trained on all rows (with small internal ES split) for artifacts/curves
#     """

#     import os
#     import json
#     import joblib
#     import numpy as np
#     import pandas as pd
#     import inspect
#     from pathlib import Path
#     from typing import Dict, Tuple, List, Optional
#     from scipy.stats import spearmanr

#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning)

#     from sklearn.model_selection import train_test_split, KFold, GroupKFold, TimeSeriesSplit
#     from sklearn.compose import ColumnTransformer
#     from sklearn.preprocessing import OneHotEncoder
#     from sklearn.pipeline import Pipeline
#     from sklearn.impute import SimpleImputer

#     from xgboost import XGBRegressor
#     import xgboost as xgb

#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     from sqlalchemy import create_engine, text
#     from dotenv import load_dotenv

#     # Optional tuning
#     try:
#         import optuna
#         HAS_OPTUNA = True
#     except Exception:
#         HAS_OPTUNA = False

#     load_dotenv()

#     THICKNESS_COLS = [
#         "Thickness Point A","Thickness Point B","Thickness Point C",
#         "Thickness Point D","Thickness Point E","Thickness Point F","Thickness Point G"
#     ]
#     ID_COLS   = ["Batch_ID"]
#     TIME_COL  = "Date_Produced"
#     TYPE_COL  = "Type"

#     # ---------------- Config ----------------
#     MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.2.4")
#     MODEL_DIR     = os.getenv("MODEL_DIR", "../models")
#     DATA_PATH     = os.getenv("TRAIN_CSV", "./data/data.csv")

#     TYPE_FILTER   = os.getenv("TYPE_FILTER", "").strip() or None
#     TARGET_CLIP_Q = tuple(float(x.strip()) for x in os.getenv("TARGET_CLIP_Q", "0.02,0.98").split(","))
#     TUNE          = int(os.getenv("TUNE", "0"))
#     ACCEPT_WNRMSE = float(os.getenv("ACCEPT_WNRMSE", "0.15"))
#     RANDOM_STATE  = int(os.getenv("RANDOM_STATE", "42"))

#     # CV
#     USE_CV   = int(os.getenv("USE_CV", "1"))
#     CV_METHOD = os.getenv("CV_METHOD", "timeseries").lower()
#     N_SPLITS  = int(os.getenv("N_SPLITS", "3"))
#     GROUP_COL = os.getenv("GROUP_COL", "Batch_ID")
#     CV_GAP    = int(os.getenv("GAP", "5"))  # purge gap in TS CV

#     # Feature handling
#     DROP_CONSTANTS     = int(os.getenv("DROP_CONSTANTS", "1"))   # drop truly constant numeric cols
#     CORR_DROP_THR      = float(os.getenv("CORR_DROP_THR", "1.0"))  # 1.0 disables corr-drop
#     SELECT_TOP_NUMERIC = int(os.getenv("SELECT_TOP_NUMERIC", "999"))
#     # Always keep critical levers + engineered features by default
#     ALWAYS_KEEP_PATTERNS = [s.strip() for s in os.getenv(
#         "ALWAYS_KEEP_PATTERNS",
#         "Axis-X_Upper_Point,Axis-Z_Upper_Point,Axis-Z_Lower_Point,"
#         "Gun1_Voltage,Gun2_Voltage,Gun1_Powder,Gun2_Powder,"
#         "Total_Powder,Total_Voltage,Exposure,Time_Index"
#     ).split(",") if s.strip()]

#     USE_EXPOSURES      = int(os.getenv("USE_EXPOSURES", "0"))
#     EXPO_SPEED_FLOOR_Q = float(os.getenv("EXPO_SPEED_FLOOR_Q", "0.20"))
#     EXPO_CLIP_QHI      = float(os.getenv("EXPO_CLIP_QHI", "0.995"))
#     EXPO_USE_LOG       = int(os.getenv("EXPO_USE_LOG", "1"))

#     # Engineered features
#     USE_FEAT_ENG   = int(os.getenv("USE_FEAT_ENG", "1"))
#     USE_TIME_INDEX = int(os.getenv("USE_TIME_INDEX", "1"))
#     TIME_INDEX_SCALE = float(os.getenv("TIME_INDEX_SCALE", "1.0"))
#     EXPOSURE_EPS   = float(os.getenv("EXPOSURE_EPS", "1e-3"))

#     # Monotonic constraints
#     USE_MONO = int(os.getenv("USE_MONO", "1"))
#     MONO_POS = [s.strip() for s in os.getenv(
#         "MONO_POS", "Voltage,Powder,total_powder,total_voltage,exposure"
#     ).split(",") if s.strip()]
#     MONO_NEG = [s.strip() for s in os.getenv(
#         "MONO_NEG", "Speed"
#     ).split(",") if s.strip()]

#     # Objective
#     XGB_OBJECTIVE = os.getenv("XGB_OBJECTIVE", "reg:pseudohubererror")  # fallback handled automatically

#     # Early stopping
#     EARLY_STOP_ROUNDS = int(os.getenv("EARLY_STOP_ROUNDS", "120"))

#     # MSSQL
#     MSSQL_DRIVER  = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
#     MSSQL_USER    = os.getenv("MSSQL_USER", "")
#     MSSQL_PASS    = os.getenv("MSSQL_PASSWORD", "")
#     MSSQL_HOST    = os.getenv("MSSQL_HOST", "(localdb)\\MSSQLLocalDB")
#     MSSQL_DB      = os.getenv("MSSQL_DB", "OptiParamAI")

#     # -------------- IO --------------
#     def make_engine():
#         drv = MSSQL_DRIVER.replace(" ", "+")
#         url = f"mssql+pyodbc://{MSSQL_USER}:{MSSQL_PASS}@{MSSQL_HOST}/{MSSQL_DB}?driver={drv}&TrustServerCertificate=yes"
#         return create_engine(url, pool_pre_ping=True, fast_executemany=True)

#     def read_table(path: str) -> pd.DataFrame:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Data file not found: {path}")
#         ext = os.path.splitext(path)[1].lower()
#         if ext in [".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"]:
#             df = pd.read_excel(path, engine="openpyxl") if ext != ".xls" else pd.read_excel(path)
#         elif ext in [".csv", ".txt"]:
#             for enc in ("utf-8","utf-8-sig","latin-1"):
#                 try:
#                     df = pd.read_csv(path, encoding=enc); break
#                 except Exception:
#                     continue
#         else:
#             df = pd.read_csv(path)
#         df.columns = [str(c).strip() for c in df.columns]
#         if TIME_COL in df.columns:
#             df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
#         return df

#     # -------------- Feature lists and selection --------------
#     def build_feature_lists(df: pd.DataFrame):
#         exclude = set(ID_COLS + [TIME_COL] + THICKNESS_COLS)
#         feat_cols = [c for c in df.columns if c not in exclude]
#         if TYPE_COL in feat_cols:
#             feat_cols = [TYPE_COL] + [c for c in feat_cols if c != TYPE_COL]
#         numeric_cols = df[feat_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
#         categorical_cols = [c for c in feat_cols if c not in numeric_cols]
#         return feat_cols, numeric_cols, categorical_cols

#     def _matches_any(name: str, patterns: List[str]) -> bool:
#         lname = name.lower()
#         return any(p.lower() in lname for p in patterns)

#     def _drop_constants(df: pd.DataFrame, cols: List[str]) -> List[str]:
#         keep = []
#         for c in cols:
#             s = df[c].dropna()
#             if s.nunique(dropna=True) <= 1:
#                 continue
#             keep.append(c)
#         return keep

#     def _drop_high_corr(df: pd.DataFrame, cols: List[str], thr: float, keep_cols: List[str]) -> List[str]:
#         if not cols or thr >= 1.0:
#             return cols
#         corr = df[cols].corr().abs()
#         upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#         to_drop = set()
#         for column in upper.columns:
#             if any(upper[column] > thr) and (column not in keep_cols):
#                 to_drop.add(column)
#         return [c for c in cols if c not in to_drop]

#     def _rank_numeric_by_spearman(df: pd.DataFrame, numeric_cols: List[str], top_n: int) -> List[str]:
#         if not numeric_cols:
#             return []
#         scores = {}
#         for col in numeric_cols:
#             vals = df[col]
#             if vals.isna().all() or vals.nunique(dropna=True) <= 1:
#                 scores[col] = 0.0; continue
#             rs = []
#             for t in THICKNESS_COLS:
#                 r, _ = spearmanr(df[col], df[t], nan_policy="omit")
#                 rs.append(abs(r) if not np.isnan(r) else 0.0)
#             scores[col] = float(np.mean(rs))
#         ranked = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
#         return ranked[:max(1, min(top_n, len(ranked)))]

#     def select_features(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
#         always_keep = [c for c in numeric_cols if _matches_any(c, ALWAYS_KEEP_PATTERNS)]
#         pruned_num = numeric_cols
#         if DROP_CONSTANTS:
#             before = len(pruned_num)
#             pruned_num = _drop_constants(df, pruned_num)
#             print(f"Dropped {before - len(pruned_num)} constant numeric features.")
#         pruned_num = _drop_high_corr(df, pruned_num, thr=CORR_DROP_THR, keep_cols=always_keep)
#         top_num = _rank_numeric_by_spearman(df, pruned_num, top_n=max(10, SELECT_TOP_NUMERIC))
#         ordered = []
#         for c in always_keep + top_num:
#             if c not in ordered:
#                 ordered.append(c)
#         feat_cols = ordered + categorical_cols
#         return feat_cols, ordered, categorical_cols, always_keep

#     # -------------- Engineered features --------------
#     def add_engineered_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
#         eng_cols = []
#         if not USE_FEAT_ENG:
#             return df, eng_cols

#         letters = list("ABCDEF")  # stations typically A..F; extend if needed
#         total_pow_all = None
#         total_volt_all = None
#         exposure_sum = None

#         for L in letters:
#             g1p = f"Rec_{L}_Gun1_Powder"
#             g2p = f"Rec_{L}_Gun2_Powder"
#             g1v = f"Rec_{L}_Gun1_Voltage"
#             g2v = f"Rec_{L}_Gun2_Voltage"
#             spd = f"Rec_{L}_Axis-Z_Speed"

#             # Total powder per station
#             if (g1p in df.columns) or (g2p in df.columns):
#                 tpow = df.get(g1p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) + \
#                     df.get(g2p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
#                 col = f"FE_{L}_Total_Powder"
#                 df[col] = tpow
#                 eng_cols.append(col)
#                 total_pow_all = (tpow if total_pow_all is None else total_pow_all + tpow)

#             # Total voltage per station
#             if (g1v in df.columns) or (g2v in df.columns):
#                 tvol = df.get(g1v, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) + \
#                     df.get(g2v, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
#                 col = f"FE_{L}_Total_Voltage"
#                 df[col] = tvol
#                 eng_cols.append(col)
#                 total_volt_all = (tvol if total_volt_all is None else total_volt_all + tvol)

#             # SAFE exposure (optional): powder / speed with floor+clip+log
#             if USE_EXPOSURES and (spd in df.columns) and ((g1p in df.columns) or (g2p in df.columns)):
#                 sp = df[spd].astype(float)
#                 # robust per-station floor; fallback to EXPOSURE_EPS if quantile is nan or tiny
#                 floor = pd.to_numeric(sp.quantile(EXPO_SPEED_FLOOR_Q), errors="coerce")
#                 if pd.isna(floor) or floor <= 0:
#                     floor = EXPOSURE_EPS
#                 sp_safe = np.maximum(sp.fillna(floor), floor)

#                 tpow = df.get(f"FE_{L}_Total_Powder", None)
#                 if tpow is None:
#                     tpow = df.get(g1p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0) + \
#                         df.get(g2p, pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)

#                 expo = tpow / sp_safe
#                 # clip extreme right tail to avoid dominating loss
#                 try:
#                     hi = float(np.nanquantile(expo.astype(float), EXPO_CLIP_QHI))
#                 except Exception:
#                     hi = np.nanmax(expo.astype(float))
#                 if not np.isfinite(hi) or hi <= 0:
#                     hi = expo.astype(float).fillna(0).quantile(0.99)
#                 expo = np.clip(expo.astype(float), 0, hi)

#                 if EXPO_USE_LOG:
#                     expo = np.log1p(expo)
#                     col = f"FE_{L}_Exposure_Log"
#                 else:
#                     col = f"FE_{L}_Exposure"

#                 df[col] = expo
#                 eng_cols.append(col)
#                 exposure_sum = (expo if exposure_sum is None else exposure_sum + expo)

#         # Global aggregates
#         if total_pow_all is not None:
#             df["FE_All_Total_Powder"] = total_pow_all; eng_cols.append("FE_All_Total_Powder")
#         if total_volt_all is not None:
#             df["FE_All_Total_Voltage"] = total_volt_all; eng_cols.append("FE_All_Total_Voltage")
#         if USE_EXPOSURES and (exposure_sum is not None):
#             df["FE_All_Exposure_Index"] = exposure_sum; eng_cols.append("FE_All_Exposure_Index")

#         # Time index
#         if USE_TIME_INDEX and (TIME_COL in df.columns) and pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
#             t0 = df[TIME_COL].min()
#             days = (df[TIME_COL] - t0).dt.total_seconds() / 86400.0
#             df["FE_Time_Index"] = days * TIME_INDEX_SCALE
#             eng_cols.append("FE_Time_Index")

#         return df, eng_cols

#     # -------------- Preprocessor --------------
#     def build_preprocessor(numeric_cols, categorical_cols):
#         num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
#         cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
#                                 ("ohe", OneHotEncoder(handle_unknown="ignore"))])
#         pre = ColumnTransformer(transformers=[
#             ("num", num_pipe, numeric_cols),
#             ("cat", cat_pipe, categorical_cols),
#         ], remainder="drop")
#         return pre

#     def _get_tx_feature_names(pre: ColumnTransformer) -> List[str]:
#         names = []
#         try:
#             names.extend(list(pre.transformers_[0][2]))  # numeric
#         except Exception:
#             pass
#         try:
#             ohe = pre.named_transformers_["cat"].named_steps["ohe"]
#             names.extend(list(ohe.get_feature_names_out()))
#         except Exception:
#             pass
#         return names

#     # -------------- Monotonic constraints --------------
#     def _mono_sign_for_name(name: str, pos: List[str], neg: List[str]) -> int:
#         lname = name.lower()
#         if any(p.lower() in lname for p in pos): return 1
#         if any(n.lower() in lname for n in neg): return -1
#         return 0

#     def build_monotone_constraints(pre: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str]) -> Optional[str]:
#         try:
#             tx_names = _get_tx_feature_names(pre)
#             num_cols = pre.transformers_[0][2] if pre.transformers_ else []
#             cons = []
#             for nm in tx_names:
#                 cons.append(_mono_sign_for_name(nm, MONO_POS, MONO_NEG) if nm in num_cols else 0)
#             return "(" + ",".join(str(int(x)) for x in cons) + ")"
#         except Exception:
#             return None

#     # -------------- XGBoost params and fitting --------------
#     def _xgb_base_params():
#         return dict(
#             n_estimators=3000,
#             learning_rate=0.035,
#             max_depth=5,
#             subsample=0.7,
#             colsample_bytree=0.7,
#             reg_lambda=4.0,
#             reg_alpha=0.5,
#             gamma=0.6,
#             min_child_weight=12,
#             tree_method="hist",
#             objective=XGB_OBJECTIVE,  # will auto-fallback if unsupported
#             eval_metric=["rmse", "mae"],
#             n_jobs=-1,
#             random_state=RANDOM_STATE,
#         )

#     # Point overrides (keep these conservative)
#     REG_OVERRIDES = {
#         "Thickness Point C": dict(max_depth=4, min_child_weight=16, subsample=0.65, colsample_bytree=0.65, reg_lambda=5.0, reg_alpha=0.6, gamma=0.8, learning_rate=0.03),
#         "Thickness Point D": dict(max_depth=4, min_child_weight=16, subsample=0.65, colsample_bytree=0.65, reg_lambda=5.0, reg_alpha=0.6, gamma=0.8, learning_rate=0.03),
#         "Thickness Point E": dict(max_depth=4, min_child_weight=16, subsample=0.65, colsample_bytree=0.65, reg_lambda=5.0, reg_alpha=0.6, gamma=0.8, learning_rate=0.03),
#         "Thickness Point F": dict(max_depth=4, min_child_weight=16, subsample=0.65, colsample_bytree=0.65, reg_lambda=5.0, reg_alpha=0.6, gamma=0.8, learning_rate=0.03),
#     }

#     def _xgb_fit_with_es(model, Xtr_t, ytr, Xva_t, yva, early_rounds: int = EARLY_STOP_ROUNDS):
#         # Try to fit; if objective unsupported, fallback to squarederror automatically
#         fit_kwargs = {"eval_set": [(Xtr_t, ytr), (Xva_t, yva)], "verbose": False}
#         sig = inspect.signature(model.fit).parameters
#         if "callbacks" in sig:
#             es = xgb.callback.EarlyStopping(rounds=early_rounds, metric_name="rmse", data_name="validation_1", save_best=True)
#             fit_kwargs["callbacks"] = [es]
#         elif "early_stopping_rounds" in sig:
#             fit_kwargs["early_stopping_rounds"] = early_rounds

#         try:
#             model.fit(Xtr_t, ytr, **fit_kwargs)
#             return model
#         except Exception as e:
#             msg = str(e).lower()
#             if ("objective" in msg) or ("pseudohuber" in msg) or ("unknown" in msg):
#                 params = model.get_params()
#                 params["objective"] = "reg:squarederror"
#                 fallback = XGBRegressor(**params)
#                 fallback.fit(Xtr_t, ytr, **fit_kwargs)
#                 return fallback
#             raise

#     def _evals_result_any(model) -> Dict:
#         for attr in ("evals_result", "evals_result_"):
#             try:
#                 er = getattr(model, attr)
#                 return er() if callable(er) else er
#             except Exception:
#                 continue
#         return {}

#     # -------------- CV splitting --------------
#     def build_ts_splits_with_gap(df: pd.DataFrame, n_splits=3, gap=0) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
#         if TIME_COL not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
#             raise ValueError("Time-series CV requires a datetime Date_Produced column.")
#         ordered = df.sort_values(TIME_COL)
#         order_idx = ordered.index.values
#         n = len(order_idx)
#         tss = TimeSeriesSplit(n_splits=n_splits)
#         splits = []
#         rel = np.arange(n)
#         for tr_rel, va_rel in tss.split(rel):
#             if gap > 0 and len(va_rel) > 0:
#                 cut = max(0, va_rel[0] - gap)
#                 tr_rel = tr_rel[tr_rel < cut]
#             tr_idx = order_idx[tr_rel]
#             va_idx = order_idx[va_rel]
#             overlap = set(tr_idx).intersection(set(va_idx))
#             assert len(overlap) == 0, f"Train/valid overlap detected: {len(overlap)}"
#             info = dict(
#                 train_rows=int(len(tr_idx)), valid_rows=int(len(va_idx)),
#                 train_start=str(ordered.loc[tr_idx, TIME_COL].min()) if len(tr_idx) else None,
#                 train_end=str(ordered.loc[tr_idx, TIME_COL].max()) if len(tr_idx) else None,
#                 valid_start=str(ordered.loc[va_idx, TIME_COL].min()) if len(va_idx) else None,
#                 valid_end=str(ordered.loc[va_idx, TIME_COL].max()) if len(va_idx) else None,
#                 gap=int(gap)
#             )
#             splits.append((tr_idx, va_idx, info))
#         return splits

#     def build_cv_splits(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
#         n = len(df)
#         if n < 10 or N_SPLITS < 2:
#             return [(np.arange(n), np.arange(0), dict(train_rows=n, valid_rows=0))]
#         if CV_METHOD == "timeseries" and TIME_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
#             return build_ts_splits_with_gap(df, n_splits=N_SPLITS, gap=CV_GAP)
#         if CV_METHOD == "group" and GROUP_COL in df.columns:
#             groups = df[GROUP_COL].astype(str).fillna("NA")
#             gkf = GroupKFold(n_splits=N_SPLITS)
#             return [(tr, va, dict(train_rows=len(tr), valid_rows=len(va))) for tr, va in gkf.split(df, groups=groups)]
#         kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
#         return [(tr, va, dict(train_rows=len(tr), valid_rows=len(va))) for tr, va in kf.split(df)]

#     # -------------- Specs / metrics --------------
#     def _collect_specs_from_df(df: pd.DataFrame) -> Dict[str,Dict]:
#         d = {}
#         for p_label, tcol in zip(list("ABCDEFG"), THICKNESS_COLS):
#             vals = df[tcol].dropna()
#             if len(vals) == 0: continue
#             lo, med, hi = float(vals.quantile(0.1)), float(vals.median()), float(vals.quantile(0.9))
#             d[p_label] = {"lower_um": lo, "upper_um": hi, "default_target_um": med, "weight": 1.0}
#         for k in ("B","C"):
#             if k in d: d[k]["weight"] = 1.2
#         return d

#     def _compute_weighted_nrmse(valid_df: pd.DataFrame, yhat: Dict[str,np.ndarray], specs: Dict[str,Dict]) -> Tuple[float, Dict[str,float]]:
#         per_point = {}
#         total_w = 0.0
#         total = 0.0
#         for p_label, tcol in zip(list("ABCDEFG"), THICKNESS_COLS):
#             y = valid_df[tcol].values
#             yh = yhat[tcol]
#             rmse = float(np.sqrt(np.mean((yh - y)**2)))
#             sp = specs.get(p_label, {})
#             lo = float(sp.get("lower_um", np.nanmin(y)))
#             hi = float(sp.get("upper_um", np.nanmax(y)))
#             span = max(1.0, hi - lo)
#             w = float(sp.get("weight", 1.0))
#             nrmse = rmse / span
#             per_point[p_label] = nrmse
#             total += w * nrmse
#             total_w += w
#         return total / max(1e-9, total_w), per_point

#     # -------------- Plots --------------
#     def _ensure_reports_dir(base_dir: str) -> Path:
#         d = Path(base_dir) / f"reports_{MODEL_VERSION}"
#         d.mkdir(parents=True, exist_ok=True)
#         return d

#     def _plot_learning_curves(curves: Dict, best_iters: Dict, outdir: Path):
#         for tcol, ev in curves.items():
#             tr = ev.get("train", [])
#             va = ev.get("valid", [])
#             if not (tr or va): continue
#             plt.figure(figsize=(8,4))
#             if tr: plt.plot(tr, label="train RMSE", color="#3b82f6")
#             if va: plt.plot(va, label="valid RMSE", color="#ef4444")
#             b = best_iters.get(tcol)
#             if b is not None and va: plt.axvline(b, color="#10b981", linestyle="--", alpha=0.8, label=f"best@{b}")
#             plt.title(f"Learning curves – {tcol}"); plt.xlabel("Boosting rounds"); plt.ylabel("RMSE")
#             plt.legend(); plt.tight_layout(); plt.savefig(outdir / f"lc_{tcol.replace(' ','_')}.png", dpi=150); plt.close()

#     def _plot_feature_importance(models: Dict, pre: ColumnTransformer, outdir: Path, topn=20):
#         tx_cols = _get_tx_feature_names(pre)
#         for tcol, model in models.items():
#             try:
#                 booster = model.get_booster()
#                 gain = booster.get_score(importance_type="gain")
#                 if not gain: continue
#                 imp = []
#                 for k, v in gain.items():
#                     idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
#                     name = tx_cols[idx] if (idx is not None and idx < len(tx_cols)) else k
#                     imp.append((name, v))
#                 imp = sorted(imp, key=lambda x: x[1], reverse=True)[:topn]
#                 plt.figure(figsize=(8, max(4, topn*0.28)))
#                 sns.barplot(x=[v for _, v in imp], y=[n for n, _ in imp], orient="h", color="#2563eb")
#                 plt.title(f"Feature importance (gain) – {tcol}")
#                 plt.xlabel("Gain"); plt.ylabel("Feature")
#                 plt.tight_layout(); plt.savefig(outdir / f"featimp_{tcol.replace(' ','_')}.png", dpi=150); plt.close()
#             except Exception:
#                 pass

#     def _plot_metrics_overview(metrics: Dict, outdir: Path):
#         pts = list(metrics.keys())
#         rmse = [metrics[p]["RMSE"] for p in pts]
#         mae  = [metrics[p]["MAE"] for p in pts]
#         plt.figure(figsize=(8,4))
#         ax = plt.gca(); ax2 = ax.twinx()
#         ax.bar(pts, rmse, color="#2563eb", alpha=0.85)
#         ax2.plot(pts, mae, color="#ef4444", marker="o")
#         ax.set_ylabel("RMSE"); ax2.set_ylabel("MAE")
#         plt.title("Validation metrics per point"); ax.grid(axis="y", alpha=0.25)
#         plt.tight_layout(); plt.savefig(outdir / "metrics_overview.png", dpi=150); plt.close()

#     def _plot_spec_norm_bars(per_point_nrmse: Dict[str,float], thr: float, outdir: Path):
#         keys = list(per_point_nrmse.keys()); vals = [per_point_nrmse[k] for k in keys]
#         plt.figure(figsize=(8,4))
#         plt.bar(keys, vals, color="#2563eb")
#         plt.axhline(thr, color="#ef4444", linestyle="--", label=f"threshold={thr}")
#         plt.ylabel("nRMSE (by spec span)"); plt.title("Spec-normalized nRMSE per point")
#         plt.legend(); plt.tight_layout(); plt.savefig(outdir / "nrmse_by_point.png", dpi=150); plt.close()

#     # -------------- DB helpers --------------
#     def upsert_types_and_specs(engine, df):
#         types = sorted(df[TYPE_COL].dropna().astype(str).unique().tolist())
#         with engine.begin() as c:
#             for tc in types:
#                 c.execute(text("""
#                     IF NOT EXISTS (SELECT 1 FROM dbo.product_type WHERE type_code = :tc)
#                         INSERT INTO dbo.product_type(type_code) VALUES (:tc);
#                 """), {"tc": tc})
#                 type_id = c.execute(text("SELECT type_id FROM dbo.product_type WHERE type_code=:tc"), {"tc": tc}).scalar()
#                 med = {p: float(df[p].median()) for p in THICKNESS_COLS}
#                 q10 = {p: float(df[p].quantile(0.10)) for p in THICKNESS_COLS}
#                 q90 = {p: float(df[p].quantile(0.90)) for p in THICKNESS_COLS}
#                 labels = list("ABCDEFG")
#                 for lab, tcol in zip(labels, THICKNESS_COLS):
#                     lo, hi, tgt = q10[tcol], q90[tcol], med[tcol]
#                     w = 1.2 if lab in ("B","C") else 1.0
#                     c.execute(text("""
#                         MERGE dbo.thickness_spec AS t
#                         USING (SELECT :tid AS type_id, :p AS point) s
#                         ON t.type_id = s.type_id AND t.point = s.point
#                         WHEN MATCHED THEN UPDATE SET lower_um=:lo, upper_um=:hi, default_target_um=:tgt, weight=:w
#                         WHEN NOT MATCHED THEN INSERT(type_id, point, lower_um, upper_um, default_target_um, weight)
#                         VALUES(:tid, :p, :lo, :hi, :tgt, :w);
#                     """), {"tid": type_id, "p": lab, "lo": lo, "hi": hi, "tgt": tgt, "w": w})

#     def compute_feature_bounds(df: pd.DataFrame, numeric_cols, top_k=16):
#         q = df[numeric_cols].quantile([0.05, 0.5, 0.95])
#         mins = df[numeric_cols].min(); maxs = df[numeric_cols].max()
#         scores = {}
#         for col in numeric_cols:
#             col_vals = df[col]
#             if col_vals.isna().all() or col_vals.nunique(dropna=True) <= 1:
#                 scores[col] = 0.0; continue
#             rs = []
#             for t in THICKNESS_COLS:
#                 r, _ = spearmanr(df[col], df[t], nan_policy="omit")
#                 rs.append(abs(r) if not np.isnan(r) else 0.0)
#             scores[col] = float(np.mean(rs))
#         top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
#         tunable = set([name for name, _ in top])
#         rows = []
#         for col in numeric_cols:
#             rows.append({
#                 "feature": col,
#                 "q05": float(q.loc[0.05, col]) if col in q.columns else float(mins[col]),
#                 "q50": float(q.loc[0.5, col]) if col in q.columns else float(mins[col]),
#                 "q95": float(q.loc[0.95, col]) if col in q.columns else float(maxs[col]),
#                 "min_val": float(mins[col]),
#                 "max_val": float(maxs[col]),
#                 "is_tunable": 1 if col in tunable else 0
#             })
#         return rows

#     def write_feature_bounds(engine, df, numeric_cols):
#         with engine.begin() as c:
#             types = sorted(df[TYPE_COL].dropna().astype(str).unique().tolist())
#             for tc in types:
#                 sub = df[df[TYPE_COL] == tc].copy()
#                 rows = compute_feature_bounds(sub, numeric_cols, top_k=16)
#                 type_id = c.execute(text("SELECT type_id FROM dbo.product_type WHERE type_code=:tc"), {"tc": tc}).scalar()
#                 c.execute(text("DELETE FROM dbo.feature_bounds WHERE type_id=:tid"), {"tid": type_id})
#                 for r in rows:
#                     c.execute(text("""
#                         INSERT INTO dbo.feature_bounds(type_id, feature, q05, q50, q95, min_val, max_val, is_tunable)
#                         VALUES(:tid, :f, :q05, :q50, :q95, :minv, :maxv, :tun)
#                     """), {"tid": type_id, "f": r["feature"], "q05": r["q05"], "q50": r["q50"],
#                         "q95": r["q95"], "minv": r["min_val"], "maxv": r["max_val"], "tun": int(r["is_tunable"])})

#     def save_artifacts(pre, models, feature_cols, numeric_cols, categorical_cols, model_dir: str):
#         os.makedirs(model_dir, exist_ok=True)
#         joblib.dump(pre, os.path.join(model_dir, "preprocessor.joblib"))
#         point_map = {
#             "Thickness Point A": "A", "Thickness Point B": "B", "Thickness Point C": "C",
#             "Thickness Point D": "D", "Thickness Point E": "E", "Thickness Point F": "F", "Thickness Point G": "G"
#         }
#         for tcol, model in models.items():
#             joblib.dump(model, os.path.join(model_dir, f"xgb_{point_map[tcol]}.joblib"))
#         info = {"feature_cols": feature_cols, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "model_version": MODEL_VERSION}
#         with open(os.path.join(model_dir, "feature_info.json"), "w") as f:
#             json.dump(info, f, indent=2)

#     def register_models(engine, metrics, model_dir: str):
#         with engine.begin() as c:
#             for tcol, m in metrics.items():
#                 point = tcol.split()[-1]
#                 uri = f"file://{os.path.abspath(os.path.join(model_dir, f'xgb_{point}.joblib'))}"
#                 c.execute(text("""
#                     INSERT INTO dbo.model_registry(model_name, version, artifact_uri, notes)
#                     VALUES(:name, :ver, :uri, :notes)
#                 """), {"name": f"xgb_{point}", "ver": MODEL_VERSION, "uri": uri, "notes": json.dumps(m)})

#     # -------------- Winsorize + holdout split --------------
#     def _winsorize_targets(df: pd.DataFrame, qlow: float, qhi: float) -> pd.DataFrame:
#         if not (0.0 <= qlow < qhi <= 1.0): return df
#         if (qlow, qhi) == (0.0, 1.0): return df
#         df2 = df.copy()
#         for t in THICKNESS_COLS:
#             if t in df2.columns:
#                 lo = df2[t].quantile(qlow); hi = df2[t].quantile(qhi)
#                 df2[t] = df2[t].clip(lower=lo, upper=hi)
#         return df2

#     def time_split(df: pd.DataFrame, test_ratio=0.1):
#         if TIME_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
#             df2 = df.sort_values(TIME_COL)
#             cut = int(len(df2) * (1 - test_ratio))
#             return df2.iloc[:cut], df2.iloc[cut:]
#         else:
#             return train_test_split(df, test_size=test_ratio, shuffle=False)

#     # -------------- CV training --------------
#     def cv_train(df: pd.DataFrame, feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str], tuned: bool = False):
#         splits = build_cv_splits(df)
#         specs = _collect_specs_from_df(df)

#         fold_metrics = []
#         fold_nrmse = []
#         fold_details = []
#         cv_curves_example = {}

#         for fold_idx, (tr_idx, va_idx, info) in enumerate(splits):
#             tr_df = df.loc[tr_idx]; va_df = df.loc[va_idx]

#             pre = build_preprocessor(numeric_cols, categorical_cols)
#             pre.fit(tr_df[feature_cols])

#             mono = build_monotone_constraints(pre, numeric_cols, categorical_cols) if USE_MONO else None

#             Xtr_t = pre.transform(tr_df[feature_cols])
#             Xva_t = pre.transform(va_df[feature_cols])

#             yhat_fold = {}
#             metrics_fold = {}
#             baseline_yhat = {}

#             for tcol in THICKNESS_COLS:
#                 ytr = tr_df[tcol].values
#                 yva = va_df[tcol].values

#                 params = _xgb_base_params()
#                 if tcol in REG_OVERRIDES: params.update(REG_OVERRIDES[tcol])
#                 if mono is not None: params["monotone_constraints"] = mono

#                 model = XGBRegressor(**params)
#                 model = _xgb_fit_with_es(model, Xtr_t, ytr, Xva_t, yva, early_rounds=EARLY_STOP_ROUNDS)
#                 yhat = model.predict(Xva_t)
#                 yhat_fold[tcol] = yhat

#                 # baseline: train median
#                 baseline_pred = np.full_like(yva, fill_value=np.median(ytr), dtype=float)
#                 baseline_yhat[tcol] = baseline_pred

#                 mae = float(np.mean(np.abs(yhat - yva)))
#                 rmse = float(np.sqrt(np.mean((yhat - yva)**2)))
#                 metrics_fold[tcol] = {"MAE": mae, "RMSE": rmse, "n_val": int(len(yva))}

#                 if fold_idx == 0:
#                     er = _evals_result_any(model)
#                     valid_rmse = er.get("validation_1", er.get("eval_1", er.get("validation", {}))).get("rmse", [])
#                     train_rmse = er.get("validation_0", er.get("eval_0", er.get("train", {}))).get("rmse", [])
#                     cv_curves_example[tcol] = {"train": train_rmse, "valid": valid_rmse, "best": (int(np.argmin(valid_rmse)) if valid_rmse else None)}

#             w_model, per_point_model = _compute_weighted_nrmse(va_df, yhat_fold, specs)
#             w_base, per_point_base  = _compute_weighted_nrmse(va_df, baseline_yhat, specs)

#             fold_nrmse.append(per_point_model | {"_weighted": w_model})
#             fold_metrics.append(metrics_fold)

#             fold_details.append({
#                 "fold": int(fold_idx+1),
#                 "info": info,
#                 "weighted_nrmse_model": float(w_model),
#                 "weighted_nrmse_baseline": float(w_base),
#                 "per_point_nrmse_model": {k: float(v) for k, v in per_point_model.items()},
#                 "per_point_nrmse_baseline": {k: float(v) for k, v in per_point_base.items()}
#             })

#         # averages
#         point_metrics_mean = {}
#         for tcol in THICKNESS_COLS:
#             r = [fm[tcol]["RMSE"] for fm in fold_metrics]
#             m = [fm[tcol]["MAE"] for fm in fold_metrics]
#             n = [fm[tcol]["n_val"] for fm in fold_metrics]
#             point_metrics_mean[tcol] = {"RMSE": float(np.mean(r)), "MAE": float(np.mean(m)), "n_val": int(np.mean(n))}
#         labels = list("ABCDEFG")
#         per_point_mean = {lab: float(np.mean([fd[lab] for fd in fold_nrmse])) for lab in labels}
#         weighted_mean = float(np.mean([fd["_weighted"] for fd in fold_nrmse]))

#         base_weighteds = [fd["weighted_nrmse_baseline"] for fd in fold_details]
#         baseline_summary = {
#             "weighted_nrmse_mean": float(np.mean(base_weighteds)),
#             "model_minus_baseline": float(weighted_mean - float(np.mean(base_weighteds)))
#         }

#         return point_metrics_mean, {"value": weighted_mean, "per_point": per_point_mean}, cv_curves_example, fold_details, baseline_summary

#     # -------------- Final fit on all rows --------------
#     def train_final_models(df: pd.DataFrame, feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str]):
#         train_df, valid_df = time_split(df, test_ratio=0.1)
#         pre = build_preprocessor(numeric_cols, categorical_cols)
#         pre.fit(train_df[feature_cols])
#         mono = build_monotone_constraints(pre, numeric_cols, categorical_cols) if USE_MONO else None
#         Xtr_t = pre.transform(train_df[feature_cols]); Xva_t = pre.transform(valid_df[feature_cols])

#         models, curves, best_iters = {}, {}, {}
#         for tcol in THICKNESS_COLS:
#             ytr, yva = train_df[tcol].values, valid_df[tcol].values
#             params = _xgb_base_params()
#             if tcol in REG_OVERRIDES: params.update(REG_OVERRIDES[tcol])
#             if mono is not None: params["monotone_constraints"] = mono
#             model = XGBRegressor(**params)
#             model = _xgb_fit_with_es(model, Xtr_t, ytr, Xva_t, yva, early_rounds=EARLY_STOP_ROUNDS)
#             models[tcol] = model
#             er = _evals_result_any(model)
#             valid_rmse = er.get("validation_1", er.get("eval_1", er.get("validation", {}))).get("rmse", [])
#             train_rmse = er.get("validation_0", er.get("eval_0", er.get("train", {}))).get("rmse", [])
#             curves[tcol] = {"train": train_rmse, "valid": valid_rmse}
#             best_iters[tcol] = int(np.argmin(valid_rmse)) if valid_rmse else None
#         return pre, models, curves, best_iters

#     # ---------------- Main ----------------
#     def main():
#         print(f"Loading data: {DATA_PATH}")
#         df = read_table(DATA_PATH)
#         if TYPE_FILTER:
#             df = df[df[TYPE_COL].astype(str) == TYPE_FILTER].copy()
#             print(f"Filtered to Type={TYPE_FILTER}  -> rows={len(df)}")

#         df = df.dropna(subset=THICKNESS_COLS, how="all")
#         if TYPE_COL not in df.columns:
#             raise ValueError("Dataset must contain 'Type' column")
#         df[TYPE_COL] = df[TYPE_COL].astype(str)

#         missing_targets = [c for c in THICKNESS_COLS if c not in df.columns]
#         if missing_targets:
#             raise KeyError(f"Missing thickness columns in data: {', '.join(missing_targets)}")

#         # Winsorize labels
#         ql, qh = TARGET_CLIP_Q
#         if (ql, qh) != (0.0, 1.0):
#             print(f"Winsorizing targets at quantiles: {ql}, {qh}")
#             df = _winsorize_targets(df, ql, qh)

#         # Add engineered features (if enabled)
#         df, eng_cols = add_engineered_features(df)
#         if eng_cols:
#             print(f"Engineered features added: {len(eng_cols)} -> {eng_cols[:10]}{' ...' if len(eng_cols)>10 else ''}")

#         # Build and select features
#         base_feat_cols, base_numeric_cols, base_categorical_cols = build_feature_lists(df)
#         print(f"Initial features: {len(base_feat_cols)} (numeric={len(base_numeric_cols)}, categorical={len(base_categorical_cols)})")
#         feat_cols, numeric_cols, categorical_cols, always_keep = select_features(df, base_numeric_cols, base_categorical_cols)
#         print(f"Selected features: {len(feat_cols)} (numeric={len(numeric_cols)}, categorical={len(categorical_cols)})")
#         print(f"Always-kept ({len(always_keep)}): {always_keep}")

#         # CV metrics
#         if USE_CV:
#             print(f"Running {CV_METHOD} cross-validation with {N_SPLITS} splits and gap={CV_GAP} ...")
#             point_metrics, wnrms_info, cv_curves, fold_details, baseline_summary = cv_train(
#                 df, feat_cols, numeric_cols, categorical_cols, tuned=bool(TUNE and HAS_OPTUNA)
#             )
#             wnrms_value = float(wnrms_info["value"])
#             per_point_nrmse = wnrms_info["per_point"]
#             print("CV metrics (avg across folds):")
#             print(json.dumps(point_metrics, indent=2))
#             print(f"Weighted normalized RMSE (CV mean): {wnrms_value:.4f}  accepted≤{ACCEPT_WNRMSE}: {wnrms_value <= ACCEPT_WNRMSE}")
#             print(f"Baseline weighted nRMSE (mean): {baseline_summary['weighted_nrmse_mean']:.4f}  "
#                 f"model - baseline: {baseline_summary['model_minus_baseline']:.4f}")
#         else:
#             # Holdout fallback
#             train_df, valid_df = time_split(df, test_ratio=0.2)
#             pre_tmp = build_preprocessor(numeric_cols, categorical_cols); pre_tmp.fit(train_df[feat_cols])
#             Xtr_t = pre_tmp.transform(train_df[feat_cols]); Xva_t = pre_tmp.transform(valid_df[feat_cols])
#             point_metrics = {}; yhat = {}
#             for tcol in THICKNESS_COLS:
#                 ytr = train_df[tcol].values; yva = valid_df[tcol].values
#                 params = _xgb_base_params()
#                 if tcol in REG_OVERRIDES: params.update(REG_OVERRIDES[tcol])
#                 model = XGBRegressor(**params)
#                 model = _xgb_fit_with_es(model, Xtr_t, ytr, Xva_t, yva, early_rounds=EARLY_STOP_ROUNDS)
#                 yh = model.predict(Xva_t); yhat[tcol] = yh
#                 mae = float(np.mean(np.abs(yh - yva))); rmse = float(np.sqrt(np.mean((yh - yva) ** 2)))
#                 point_metrics[tcol] = {"MAE": mae, "RMSE": rmse, "n_val": int(len(yva))}
#             specs = _collect_specs_from_df(df)
#             wnrms_value, per_point_nrmse = _compute_weighted_nrmse(valid_df, yhat, specs)
#             cv_curves = {}; fold_details = []; baseline_summary = {}

#         # Final fit for artifacts
#         print("Training final models on all rows (with internal ES split) ...")
#         pre, models, curves, best_iters = train_final_models(df, feat_cols, numeric_cols, categorical_cols)

#         outdir = Path(MODEL_DIR) / f"reports_{MODEL_VERSION}"; outdir.mkdir(parents=True, exist_ok=True)
#         _plot_learning_curves({k: {"train": v["train"], "valid": v["valid"]} for k, v in curves.items()}, best_iters, outdir)
#         _plot_feature_importance(models, pre, outdir, topn=20)
#         _plot_metrics_overview(point_metrics, outdir)
#         _plot_spec_norm_bars(per_point_nrmse, ACCEPT_WNRMSE, outdir)

#         # Persist selected features and engineering info
#         with open(outdir / "selected_features.json", "w") as f:
#             json.dump({
#                 "numeric_cols": numeric_cols,
#                 "categorical_cols": categorical_cols,
#                 "all_feature_cols": feat_cols,
#                 "always_kept": always_keep,
#                 "engineered_features": eng_cols,
#                 "use_time_index": bool(USE_TIME_INDEX)
#             }, f, indent=2)

#         # Artifacts + DB
#         save_artifacts(pre, models, feat_cols, numeric_cols, categorical_cols, MODEL_DIR)
#         eng = make_engine()
#         upsert_types_and_specs(eng, df)
#         write_feature_bounds(eng, df, numeric_cols)
#         register_models(eng, point_metrics, MODEL_DIR)

#         # Report
#         report = {
#             "model_version": MODEL_VERSION,
#             "type_filter": TYPE_FILTER,
#             "rows": {"total": int(len(df))},
#             "features": {"total": len(feat_cols), "numeric": len(numeric_cols), "categorical": len(categorical_cols)},
#             "metrics": point_metrics,  # CV-mean
#             "weighted_nrmse": {
#                 "value": float(wnrms_value),
#                 "per_point": {k: float(v) for k, v in per_point_nrmse.items()},
#                 "accept_threshold": ACCEPT_WNRMSE,
#                 "accepted": bool(wnrms_value <= ACCEPT_WNRMSE)
#             },
#             "cv": {
#                 "used": bool(USE_CV),
#                 "method": CV_METHOD,
#                 "n_splits": N_SPLITS,
#                 "gap": CV_GAP,
#                 "folds": fold_details
#             },
#             "baseline": baseline_summary,
#             "tuning": {"used": bool(TUNE and HAS_OPTUNA)},
#             "notes": {
#                 "feature_pruning": {
#                     "drop_constants": bool(DROP_CONSTANTS),
#                     "corr_drop_thr": CORR_DROP_THR,
#                     "select_top_numeric": SELECT_TOP_NUMERIC,
#                     "always_keep_patterns": ALWAYS_KEEP_PATTERNS
#                 },
#                 "monotonic_constraints": {"used": bool(USE_MONO), "pos_keywords": MONO_POS, "neg_keywords": MONO_NEG},
#                 "winsorization": {"qlow": ql, "qhi": qh},
#                 "engineered": {"used": bool(USE_FEAT_ENG), "created": eng_cols, "use_time_index": bool(USE_TIME_INDEX)},
#                 "objective": XGB_OBJECTIVE
#             }
#         }
#         with open(outdir / "training_report.json", "w") as f:
#             json.dump(report, f, indent=2)
#         print(f"Artifacts and training report saved to {outdir}")

#     if __name__ == "__main__":
#         main()
# else:
#  none