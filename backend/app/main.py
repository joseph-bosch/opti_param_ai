import orjson
import uuid
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
import random
import string

from .logging_utils import setup_logging, get_logger, request_id_ctx, run_id_ctx
from .db import ENGINE
from .model_service import ModelService, POINTS
from .specs_service import (
    get_types, get_specs, get_feature_bounds,
    upsert_type, normalize_type_code, ensure_product_type_uniqueness
)
from .optimizer_service import optimize
from .api_schemas import (
    PredictRequest, PredictResponse,
    RecommendRequest, RecommendResponse,
    TypeUpsertRequest, TypeUpsertResponse,
)
from .config import MODEL_VERSION
from pydantic import BaseModel

setup_logging()
log = get_logger("api")

app = FastAPI(title="OptiParam AI API", version=MODEL_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


ensure_product_type_uniqueness()

ROUNDING_RULES = [
    ("_Powder", 0),
    ("_Total_Air", 1),
    ("_Voltage", 0),
    ("_Current", 0),
    ("_Electrode_Cleaning", 1),
    ("onoff_before_after_obj", 0),
    ("_Upper_Point", 0),
    ("_Lower_Point", 0),
    ("_Speed", 0),
    ("_Spray_Dist", 0),
    ("Spray_onoff_before_after", 0),
]

def quantize_feature_value(name: str, value: float) -> float:
    if value is None:
        return value
    for token, decimals in ROUNDING_RULES:
        if token in name:
            factor = 10 ** decimals
            return round(float(value) * factor) / factor
    return round(float(value), 2)

def quantize_recommended(rec: dict[str, float]) -> dict[str, float]:
    return {k: quantize_feature_value(k, v) for k, v in rec.items()}

def make_trial_code(length: int = 4) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_ctx.set(rid)
    try:
        log.info("request.start method=%s path=%s", request.method, request.url.path)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info("request.end status=%s", response.status_code)
        return response
    finally:
        request_id_ctx.reset(token)

model_service = ModelService()
MODELS_LOADED = model_service.load()
log.info("models.loaded count=%s version=%s", MODELS_LOADED, MODEL_VERSION)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED, "version": MODEL_VERSION}

class TrialStartResponse(BaseModel):
    trial_code: str

@app.post("/trial/start", response_model=TrialStartResponse)
def start_trial(response: Response):
    run_token = run_id_ctx.set(str(uuid.uuid4()))
    response.headers["X-Run-ID"] = run_id_ctx.get()
    try:
        code = make_trial_code()

        with ENGINE.begin() as c:
            c.execute(
                text("""
                    INSERT INTO dbo.recommendation_log
                        (type_code, targets_json, recommendation_json,
                         objective_score, trials, model_version, trial_code)
                    VALUES (:tc, :t, :r, :s, :n, :v, :trial_code)
                """),
                {
                    "tc": "TRIAL_START",
                    "t": orjson.dumps({}).decode(),
                    "r": orjson.dumps({}).decode(),
                    "s": 0.0,
                    "n": 0,
                    "v": MODEL_VERSION,
                    "trial_code": code,
                },
            )

        log.info("trial.start code=%s", code)
        return TrialStartResponse(trial_code=code)
    finally:
        run_id_ctx.reset(run_token)

@app.get("/types")
def types():
    return {"types": get_types()}

@app.post("/types", response_model=TypeUpsertResponse)
def types_upsert(req: TypeUpsertRequest, response: Response):
    """
    Upsert product type by type_code (trim+upper). Returns type_id always.
    """
    run_token = run_id_ctx.set(str(uuid.uuid4()))
    response.headers["X-Run-ID"] = run_id_ctx.get()
    try:
        tc = normalize_type_code(req.type_code)
        if not tc:
            raise HTTPException(status_code=400, detail="type_code is required")

        type_id, created, norm_code = upsert_type(tc, req.description)
        log.info("type.upsert type_code=%s type_id=%s created=%s", norm_code, type_id, created)
        return TypeUpsertResponse(type_id=type_id, type_code=norm_code, created=created)
    finally:
        run_id_ctx.reset(run_token)

@app.get("/specs/{type_code}")
def specs(type_code: str):
    tc = normalize_type_code(type_code)
    return {"type_code": tc, "specs": get_specs(tc)}

@app.get("/feature-bounds/{type_code}")
def feature_bounds(type_code: str):
    tc = normalize_type_code(type_code)
    return {"type_code": tc, "bounds": get_feature_bounds(tc)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, response: Response):
    run_token = run_id_ctx.set(str(uuid.uuid4()))
    response.headers["X-Run-ID"] = run_id_ctx.get()
    try:
        tc = normalize_type_code(req.type_code)
        if req.features is None:
            req.features = {}
        req.features.setdefault("Type", tc)

        log.info("predict.begin type=%s features=%s", tc, list((req.features or {}).keys()))
        specs = get_specs(tc)
        preds = model_service.predict(tc, req.features or {})

        in_spec, margins = {}, {}
        for p in POINTS:
            y = preds.get(p)
            if y is None:
                in_spec[p] = None
                margins[p] = None
            else:
                lo, hi = specs.get(p, {}).get("lo"), specs.get(p, {}).get("hi")
                if lo is None or hi is None:
                    in_spec[p] = None
                    margins[p] = None
                else:
                    in_spec[p] = (lo <= y <= hi)
                    margins[p] = min(y - lo, hi - y)

        with ENGINE.begin() as c:
            c.execute(
                text("""
                    INSERT INTO dbo.prediction_log(type_code, inputs_json, predictions_json, model_version, trial_code)
                    VALUES (:tc, :i, :o, :v, :trial_code)
                """),
                {
                    "tc": tc,
                    "i": orjson.dumps({"features": req.features or {}, "run_id": run_id_ctx.get()}).decode(),
                    "o": orjson.dumps(preds).decode(),
                    "v": MODEL_VERSION,
                    "trial_code": req.trial_code,
                },
            )

        log.info("predict.end")
        return PredictResponse(
            predictions=preds,
            in_spec=in_spec,
            margins=margins,
            model_version=MODEL_VERSION,
        )
    finally:
        run_id_ctx.reset(run_token)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, response: Response):
    run_token = run_id_ctx.set(str(uuid.uuid4()))
    response.headers["X-Run-ID"] = run_id_ctx.get()
    try:
        tc = normalize_type_code(req.type_code)
        log.info("recommend.begin type=%s context_keys=%s n_trials=%s",
                 tc, list((req.fixed_context or {}).keys()), req.n_trials)

        specs = get_specs(tc)
        if req.targets:
            for p, t in req.targets.items():
                if p in specs:
                    specs[p]["target"] = float(t)

        best_params, predicted, score, trials, _ = optimize(
            model_service=model_service,
            type_code=tc,
            specs=specs,
            fixed_context=req.fixed_context or {},
            n_trials=req.n_trials,
            timeout=req.timeout_sec,
            current=req.current,
            step_pct=req.step_pct,
        )

        rounded_params = quantize_recommended(best_params)

        with ENGINE.begin() as c:
            c.execute(
                text("""
                    INSERT INTO dbo.recommendation_log
                    (type_code, targets_json, recommendation_json, objective_score, trials, model_version, trial_code)
                    VALUES (:tc, :t, :r, :s, :n, :v, :trial_code)
                """),
                {
                    "tc": tc,
                    "t": orjson.dumps(req.targets or {}).decode(),
                    "r": orjson.dumps({"recommended": best_params, "run_id": run_id_ctx.get()}).decode(),
                    "s": float(score),
                    "n": int(trials),
                    "v": MODEL_VERSION,
                    "trial_code": req.trial_code,
                },
            )

        log.info("recommend.end trials=%s score=%.4f", trials, score)
        return RecommendResponse(
            recommended=rounded_params,
            predicted=predicted,
            score=score,
            trials=trials,
            model_version=MODEL_VERSION,
        )
    finally:
        run_id_ctx.reset(run_token)
