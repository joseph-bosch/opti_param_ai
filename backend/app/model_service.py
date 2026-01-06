import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from .config import MODEL_DIR
from .specs_service import get_feature_bounds
from .logging_utils import get_logger

log = get_logger("model_service")

POINTS = list("ABCDEFG")

PREPROCESSOR_NAME = "preprocessor.joblib"
CAT_MAPPINGS = "cat_mappings.json"
FILL_DEFAULTS = "fill_defaults.json"
STACK_INFO = "stack_info.json"  

BASE_MODEL_PATTERNS = [
    "{algo}_Thickness_Point_{p}.joblib",
    "{algo}_{p}.joblib",
    "{algo}_Thickness_Point_{p}.tf",
    "{algo}_{p}.tf",
]

STACK_MODEL_PATTERNS = [
    "stack_meta_Thickness_Point_{p}.joblib",
    "stack_meta_{p}.joblib",
]


REC_PREFIX = "Rec_"
FE_PREFIX = "FE_"


FE_TIME_INDEX_DEFAULT = 0.0


PREDICT_DEBUG = int(os.getenv("PREDICT_DEBUG", "1"))
RETURN_DEBUG = int(os.getenv("RETURN_DEBUG", "0"))


# ------------------------- LOW LEVEL HELPERS -------------------------


def _load_joblib(path: str):
    try:
        return joblib.load(path)
    except Exception:
        log.exception("joblib.load_failed path=%s", path)
        return None


def _load_tf(path: str):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except Exception:
        log.exception("tf.load_failed path=%s", path)
        return None


def _to_float_safe(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        return float(str(x).strip())
    except Exception:
        return 0.0


def _sum_keys(d: Dict[str, Any], must_contain: List[str], letter: Optional[str] = None) -> float:
    """Sum over Rec_* features matching all substrings; optionally restrict to a letter A..G."""
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
    Recompute engineered FE_* features from Rec_* + context.
    Only affects inference-time feature values; not part of the search space.
    """
    fe: Dict[str, float] = {}


    fe["FE_All_Total_Voltage"] = _sum_keys(all_feats, ["_Voltage"])
    fe["FE_All_Total_Powder"] = _sum_keys(all_feats, ["_Powder"])


    fe["FE_C_Total_Voltage"] = _sum_keys(all_feats, ["_Voltage"], letter="C")
    fe["FE_D_Total_Voltage"] = _sum_keys(all_feats, ["_Voltage"], letter="D")


    if "FE_Time_Index" in all_feats:
        fe["FE_Time_Index"] = _to_float_safe(all_feats["FE_Time_Index"])
    else:
        fe["FE_Time_Index"] = FE_TIME_INDEX_DEFAULT

    return fe


# ------------------------- MODEL SERVICE -------------------------


class ModelService:
    """
    Prediction service using base models (XGB / CatBoost / LGBM / NN)
    stacked by a meta-model per thickness point.

    If a stack_meta_* model is missing for a given point, we fall back to a
    weighted blend of the available base models (XGB / CatBoost / LGBM).
    """

    BASE_ALGOS = ("xgb", "cat", "lgb", "nn")

    def __init__(self):

        self.pre = None  
        self.cat_mappings: Dict[str, List[str]] = {}
        self.fill_defaults: Dict[str, Any] = {}
        self.features: List[str] = []    
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []

        self.models: Dict[str, Dict[str, Any]] = {algo: {} for algo in self.BASE_ALGOS}

  
        self.stack_models: Dict[str, Any] = {}
        self.stack_feature_names: Dict[str, List[str]] = {}

        self._loaded = False
        try:
            self.load()
        except Exception:
            log.exception("ModelService.__init__ load failed")

    # ------------------------- LOAD ARTIFACTS -------------------------

    def _try_load_base_model(self, algo: str, p: str):
        for pat in BASE_MODEL_PATTERNS:
            fp = os.path.join(MODEL_DIR, pat.format(algo=algo, p=p))
            if os.path.exists(fp):
                m = _load_tf(fp) if fp.endswith(".tf") else _load_joblib(fp)
                if m is not None:
                    log.info("Loaded %s model for point %s from %s", algo, p, fp)
                    return m
        return None

    def _try_load_stack_model(self, p: str):
        for pat in STACK_MODEL_PATTERNS:
            fp = os.path.join(MODEL_DIR, pat.format(p=p))
            if os.path.exists(fp):
                m = _load_joblib(fp)
                if m is not None:
                    self.stack_models[p] = m
                    fn = list(getattr(m, "feature_names_in_", []) or [])
                    self.stack_feature_names[p] = fn
                    log.info(
                        "Loaded stack_meta model for point %s from %s (n_features=%d)",
                        p,
                        fp,
                        len(fn),
                    )
                    return True
        return False

    def _load_stack_info(self):
        path = os.path.join(MODEL_DIR, STACK_INFO)
        if not os.path.exists(path):
            log.warning("stack_info.json not found; falling back to preprocessor inputs / cat_mappings.")
            return
        try:
            with open(path, "r") as fh:
                info = json.load(fh)
            self.features = list(info.get("features", []) or [])
            self.numeric_cols = list(info.get("numeric_cols", []) or [])

            self.categorical_cols = list(
                info.get("categororical_cols", info.get("categorical_cols", []) or [])
            )
            log.info(
                "Loaded stack_info: %d features (num=%d, cat=%d)",
                len(self.features), len(self.numeric_cols), len(self.categorical_cols)
            )
        except Exception:
            log.exception("stack_info.load_failed")

    def load(self) -> int:
        loaded = 0


        pre_path = os.path.join(MODEL_DIR, PREPROCESSOR_NAME)
        if os.path.exists(pre_path):
            try:
                self.pre = joblib.load(pre_path)
                log.info("Loaded preprocessor from %s", pre_path)
            except Exception:
                log.exception("preprocessor.load_failed")


        cat_path = os.path.join(MODEL_DIR, CAT_MAPPINGS)
        if os.path.exists(cat_path):
            try:
                with open(cat_path, "r") as fh:
                    self.cat_mappings = json.load(fh)
                log.info("Loaded cat_mappings for cols=%s", list(self.cat_mappings.keys()))
            except Exception:
                log.exception("cat_mappings.load_failed")


        fill_path = os.path.join(MODEL_DIR, FILL_DEFAULTS)
        if os.path.exists(fill_path):
            try:
                with open(fill_path, "r") as fh:
                    self.fill_defaults = json.load(fh)
                log.info("Loaded fill_defaults for %d cols", len(self.fill_defaults))
            except Exception:
                log.exception("fill_defaults.load_failed")

        self._load_stack_info()


        if not self.features and self.pre is not None and hasattr(self.pre, "feature_names_in_"):
            self.features = list(getattr(self.pre, "feature_names_in_"))
        if not self.categorical_cols and self.cat_mappings:
            self.categorical_cols = list(self.cat_mappings.keys())
            self.numeric_cols = [c for c in self.features if c not in self.categorical_cols]

        # Base models
        for algo in self.BASE_ALGOS:
            for p in POINTS:
                m = self._try_load_base_model(algo, p)
                if m is not None:
                    self.models[algo][p] = m
                    loaded += 1

        # Stack meta models
        stack_loaded = 0
        for p in POINTS:
            if self._try_load_stack_model(p):
                stack_loaded += 1

        self._loaded = True
        log.info(
            "Loaded %d base models across algos=%s and %d stack_meta models",
            loaded,
            list(self.BASE_ALGOS),
            stack_loaded,
        )
        return loaded + stack_loaded

    # ------------------------- FEATURE CONSTRUCTION -------------------------

    def _complete_features(self, type_code: str, user_feats: Dict[str, Any]) -> Dict[str, Any]:
        try:
            _ = get_feature_bounds(type_code) or {}
        except Exception:

            pass
        full = dict(user_feats or {})
        full.setdefault("Type", type_code)
        return full

    def _coerce_numeric(self, s: pd.Series, col: str) -> pd.Series:
        default_val = self.fill_defaults.get(col, 0.0)
        return pd.to_numeric(s, errors="coerce").fillna(default_val)

    def _apply_cat_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_cols:
            return df
        out = df.copy()
        for col in self.categorical_cols:
            if col not in out.columns:
                continue
            cats = None
            if self.cat_mappings.get(col):
                cats = [str(c) for c in self.cat_mappings[col]]
                if "__NA__" not in cats:
                    cats.append("__NA__")
            out[col] = out[col].astype(str).fillna("__NA__")
            try:
                out[col] = (
                    pd.Categorical(out[col], categories=cats, ordered=False)
                    if cats
                    else out[col].astype("category")
                )
            except Exception:
                out[col] = out[col].astype("category")
        return out

    def _construct_tree_dataframe(
        self,
        type_code: str,
        user_feats: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:

        if not self.features:
            raise RuntimeError("No training feature list available (stack_info.json missing).")

        base = self._complete_features(type_code, user_feats)

        # Single-pass construction to avoid fragmentation; start with raw row values
        row: Dict[str, Any] = {}
        provided_set = set(base.keys())
        feature_set = set(self.features)

        missing_for_user = [c for c in self.features if c not in base]
        overlap = len(provided_set & feature_set)

        for col in self.features:
            if col in base:
                val = base[col]
            else:
                val = "__NA__" if col in self.categorical_cols else self.fill_defaults.get(col, 0.0)
            row[col] = val

        fe_vals = _derive_engineered_features(row)
        fe_applied = []
        for k, v in fe_vals.items():
            if k in self.features:
                row[k] = v
                fe_applied.append(k)


        X = pd.DataFrame([row], columns=self.features)


        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = self._coerce_numeric(X[col], col)

        # Categoricals
        X = self._apply_cat_dtype(X)

        diag = {
            "n_training_features": len(self.features),
            "n_user_provided": len(provided_set),
            "n_overlap_provided": overlap,
            "n_missing_filled_defaults": len(missing_for_user),
            "missing_filled_examples": missing_for_user[:10],
            "fe_recomputed": fe_applied[:10],
        }
        return X, diag

    # ------------------------- INTERNAL TRANSFORMS -------------------------

    @staticmethod
    def _encode_cats_for_xgb(df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for col in X.columns:
            if str(X[col].dtype) == "category":
                codes = X[col].cat.codes.replace({-1: np.nan})
                X[col] = codes
        return X

    @staticmethod
    def _align_to_feature_names(
        X: pd.DataFrame,
        names: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        if not names:
            return X, [], []
        extra = [c for c in X.columns if c not in names]
        if extra:
            X = X.drop(columns=extra)
        missing = [c for c in names if c not in X.columns]
        for m in missing:
            X[m] = np.nan
        return X[names], missing, extra

    def _build_nn_input(self, X_tree: pd.DataFrame) -> np.ndarray:
        """
        Build NN input array, preferring the saved preprocessor if available.
        """
        if self.pre is not None:
            try:
                
                feat_names = getattr(self.pre, "feature_names_in_", None)
                if feat_names is None:
                    names = []
                else:
                    names = [str(c) for c in list(feat_names)]


                X_in, _, _ = self._align_to_feature_names(X_tree.copy(), names or list(X_tree.columns))
                return self.pre.transform(X_in)
            except Exception:
                log.exception("preprocessor.transform_failed; falling back to numeric-only NN input")

        # Fallback: numeric columns only
        return X_tree.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    # ------------------------- PER-ALGO PREDICTORS -------------------------

    def _predict_xgb_for_point(
        self,
        p: str,
        X_xgb_raw: pd.DataFrame,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        dbg: Dict[str, Any] = {"xgb_used": False}
        if p not in self.models["xgb"]:
            return None, dbg
        try:
            model_x = self.models["xgb"][p]
            
            feat_names = getattr(model_x, "feature_names_in_", None)


            if feat_names is None:

                names = list(X_xgb_raw.columns)
            else:

                names = [str(c) for c in list(feat_names)]

                if len(names) == 0:
                    names = list(X_xgb_raw.columns)
            

            Xx, missing, extra = self._align_to_feature_names(X_xgb_raw.copy(), names)
            prx = model_x.predict(Xx)
            val = float(prx[0]) if prx is not None and len(prx) > 0 and not np.isnan(prx[0]) else None
            dbg.update(
                {
                    "xgb_used": True,
                    "xgb_pred": val,
                    "xgb_missing_after_align": missing[:10],
                    "xgb_n_missing_after_align": len(missing),
                    "xgb_n_extra_dropped": len(extra),
                }
            )
            return val, dbg
        except Exception:
            log.exception("xgb_predict_failed for %s", p)
            dbg.update({"xgb_used": True, "xgb_error": True})
            return None, dbg

    def _predict_cat_for_point(
        self,
        p: str,
        X_tree: pd.DataFrame,
        cat_pool,
        cat_feat_idxs: List[int],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        dbg: Dict[str, Any] = {"cat_used": False}
        if p not in self.models["cat"]:
            return None, dbg
        try:
            model_c = self.models["cat"][p]
            if cat_pool is not None:
                prc = model_c.predict(cat_pool)
            else:
                prc = model_c.predict(self._apply_cat_dtype(X_tree.copy()))
            val = float(prc[0]) if prc is not None and len(prc) > 0 and not np.isnan(prc[0]) else None
            dbg.update(
                {
                    "cat_used": True,
                    "cat_pred": val,
                    "cat_feat_indices_count": len(cat_feat_idxs),
                }
            )
            return val, dbg
        except Exception:
            log.exception("cat_predict_failed for %s", p)
            dbg.update({"cat_used": True, "cat_error": True})
            return None, dbg

    def _predict_lgb_for_point(
        self,
        p: str,
        X_tree: pd.DataFrame,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        dbg: Dict[str, Any] = {"lgb_used": False}
        if p not in self.models["lgb"]:
            return None, dbg
        try:
            model_l = self.models["lgb"][p]

            names = list(
                getattr(model_l, "feature_name_", [])
                or getattr(model_l, "feature_names_in_", [])
                or list(X_tree.columns)
            )
            Xl, missing, extra = self._align_to_feature_names(X_tree.copy(), names)
            prl = model_l.predict(Xl)
            val = float(prl[0]) if prl is not None and len(prl) > 0 and not np.isnan(prl[0]) else None
            dbg.update(
                {
                    "lgb_used": True,
                    "lgb_pred": val,
                    "lgb_missing_after_align": missing[:10],
                    "lgb_n_missing_after_align": len(missing),
                    "lgb_n_extra_dropped": len(extra),
                }
            )
            return val, dbg
        except Exception:
            log.exception("lgb_predict_failed for %s", p)
            dbg.update({"lgb_used": True, "lgb_error": True})
            return None, dbg

    def _predict_nn_for_point(
        self,
        p: str,
        X_tree: pd.DataFrame,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        dbg: Dict[str, Any] = {"nn_used": False}
        if p not in self.models["nn"]:
            return None, dbg
        try:
            model_n = self.models["nn"][p]
            Xn = self._build_nn_input(X_tree)
            prn = model_n.predict(Xn)

            if isinstance(prn, (list, tuple)):
                prn = np.asarray(prn)
            prn = np.ravel(prn)
            val = float(prn[0]) if len(prn) > 0 and not np.isnan(prn[0]) else None
            dbg.update(
                {
                    "nn_used": True,
                    "nn_pred": val,
                    "nn_input_shape": getattr(Xn, "shape", None),
                }
            )
            return val, dbg
        except Exception:
            log.exception("nn_predict_failed for %s", p)
            dbg.update({"nn_used": True, "nn_error": True})
            return None, dbg

    def _predict_stack_for_point(
        self,
        p: str,
        base_preds: Dict[str, Optional[float]],
        X_tree: pd.DataFrame,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        dbg: Dict[str, Any] = {"stack_used": False}
        if p not in self.stack_models:
            return None, dbg
        model_s = self.stack_models[p]
        names = self.stack_feature_names.get(p) or list(
            getattr(model_s, "feature_names_in_", []) or []
        )


        base_map: Dict[str, Optional[float]] = {}
        for algo in ("xgb", "cat", "lgb", "nn"):
            v = base_preds.get(algo)
            if v is None:
                continue
            base_map[algo] = v
            base_map[f"{algo}_pred"] = v
            base_map[f"pred_{algo}"] = v

        unknown_features: List[str] = []

        if names:
            row = {}
            for f in names:
                if f in base_map:
                    row[f] = base_map[f]
                elif f in X_tree.columns:
                    row[f] = _to_float_safe(X_tree[f].iloc[0])
                else:
                    row[f] = 0.0
                    unknown_features.append(f)
            Xs = pd.DataFrame([row], columns=names)
        else:

            vals = [base_preds[a] for a in ("xgb", "cat", "lgb", "nn") if base_preds.get(a) is not None]
            if not vals:
                return None, dbg
            Xs = np.array([vals], dtype=float)

        try:
            prs = model_s.predict(Xs)
            if isinstance(prs, (list, tuple)):
                prs = np.asarray(prs)
            prs = np.ravel(prs)
            val = float(prs[0]) if len(prs) > 0 and not np.isnan(prs[0]) else None
            dbg.update(
                {
                    "stack_used": True,
                    "stack_pred": val,
                    "stack_feature_names": names,
                    "stack_unknown_features_filled_zero": unknown_features[:10],
                }
            )
            return val, dbg
        except Exception:
            log.exception("stack_meta_predict_failed for %s", p)
            dbg.update({"stack_used": True, "stack_error": True})
            return None, dbg

    # ------------------------- MAIN PREDICT -------------------------

    def predict(self, type_code: str, user_feats: Dict[str, Any]) -> Dict[str, float]:
        if not self._loaded:
            log.warning("predict called but not loaded")
            return {p: None for p in POINTS}


        try:
            X_tree, diag = self._construct_tree_dataframe(type_code, user_feats)
        except Exception:
            log.exception("failed to construct tree dataframe")
            return {p: None for p in POINTS}


        try:
            from catboost import Pool as CatPool
            cat_feat_idxs = [X_tree.columns.get_loc(c) for c in self.categorical_cols if c in X_tree.columns]
            cat_pool = CatPool(X_tree, cat_features=cat_feat_idxs) if len(cat_feat_idxs) > 0 else CatPool(X_tree)
        except Exception:
            cat_pool = None
            cat_feat_idxs = []
            log.exception("catboost.Pool creation failed")


        X_xgb_raw = self._encode_cats_for_xgb(X_tree)

   
        debug_global = {
            "type": type_code,
            **diag,
            "categorical_cols_count": len(self.categorical_cols),
            "xgb_models_loaded": sorted(list(self.models["xgb"].keys())),
            "cat_models_loaded": sorted(list(self.models["cat"].keys())),
            "lgb_models_loaded": sorted(list(self.models["lgb"].keys())),
            "nn_models_loaded": sorted(list(self.models["nn"].keys())),
            "stack_models_loaded": sorted(list(self.stack_models.keys())),
            "cat_feat_indices_count": len(cat_feat_idxs),
        }

        result: Dict[str, Any] = {}
        if RETURN_DEBUG:
            result["_debug"] = {"global": debug_global, "points": {}}


        fallback_weights = {"xgb": 0.7, "cat": 0.2, "lgb": 0.1}

        for p in POINTS:
            point_dbg: Dict[str, Any] = {}

            # --- base predictions ---
            pred_xgb, dbg_xgb = self._predict_xgb_for_point(p, X_xgb_raw)
            pred_cat, dbg_cat = self._predict_cat_for_point(p, X_tree, cat_pool, cat_feat_idxs)
            pred_lgb, dbg_lgb = self._predict_lgb_for_point(p, X_tree)
            pred_nn, dbg_nn = self._predict_nn_for_point(p, X_tree)

            base_preds = {
                "xgb": pred_xgb,
                "cat": pred_cat,
                "lgb": pred_lgb,
                "nn": pred_nn,
            }


            point_dbg.update(dbg_xgb)
            point_dbg.update(dbg_cat)
            point_dbg.update(dbg_lgb)
            point_dbg.update(dbg_nn)

            # --- stack meta prediction (preferred path) ---
            pred_stack, dbg_stack = self._predict_stack_for_point(p, base_preds, X_tree)
            point_dbg.update(dbg_stack)

            if pred_stack is not None:
                final_pred = pred_stack
                point_dbg["final_source"] = "stack_meta"
            else:

                num = 0.0
                den = 0.0
                for algo, w in fallback_weights.items():
                    v = base_preds.get(algo)
                    if v is None:
                        continue
                    num += w * v
                    den += w
                final_pred = num / den if den > 0 else None
                point_dbg["final_source"] = "fallback_blend"
                point_dbg["fallback_pred"] = final_pred

            result[p] = final_pred

            if PREDICT_DEBUG or RETURN_DEBUG:
                if RETURN_DEBUG:
                    result["_debug"]["points"][p] = point_dbg
                else:
                    log.info("[predict-debug] point=%s details=%s", p, point_dbg)


        if PREDICT_DEBUG and not RETURN_DEBUG:
            log.info("[predict-debug] global=%s", debug_global)


        return result

    # ------------------------- ADMIN -------------------------

    def list_models(self):
        return {
            "preprocessor_loaded": self.pre is not None,
            "xgb_points": list(self.models["xgb"].keys()),
            "cat_points": list(self.models["cat"].keys()),
            "lgb_points": list(self.models["lgb"].keys()),
            "nn_points": list(self.models["nn"].keys()),
            "stack_points": list(self.stack_models.keys()),
            "cat_mappings": list(self.cat_mappings.keys()),
            "fill_defaults_loaded": len(self.fill_defaults),
            "features_count": len(self.features),
            "numeric_cols_count": len(self.numeric_cols),
            "categorical_cols_count": len(self.categorical_cols),
        }

    def reload(self):
        self.__init__()
        return self.list_models()


