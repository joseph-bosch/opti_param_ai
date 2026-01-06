
from pydantic import BaseModel, Field, conint, confloat
from typing import Dict, Optional, Union

NumberOrStr = Union[float, int, str]


class PredictRequest(BaseModel):
    type_code: str
    features: Dict[str, NumberOrStr] = Field(default_factory=dict)
    trial_code: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: Dict[str, Optional[float]]
    in_spec: Dict[str, Optional[bool]]
    margins: Dict[str, Optional[float]]
    model_version: str


class RecommendRequest(BaseModel):
    type_code: str
    targets: Dict[str, float] | None = None
    fixed_context: Dict[str, NumberOrStr] = Field(default_factory=dict)

    current: Dict[str, float] | None = None
    step_pct: confloat(ge=0.0, le=0.2) = 0.02
    n_trials: conint(ge=1, le=2000) = 150
    timeout_sec: Optional[confloat(gt=0)] = 5
    trial_code: Optional[str] = None


class RecommendResponse(BaseModel):
    recommended: Dict[str, float]
    predicted: Dict[str, float]
    score: float
    trials: int
    model_version: str


# ---------------------------
# NEW: Product type upsert
# ---------------------------

class TypeUpsertRequest(BaseModel):
    type_code: str
    description: Optional[str] = None


class TypeUpsertResponse(BaseModel):
    type_id: int
    type_code: str
    created: bool
