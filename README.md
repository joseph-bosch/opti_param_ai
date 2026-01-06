# OptiParam AI System

A production-oriented system for **parameter optimization** and **thickness prediction (A–G points)** that combines:

- **Frontend**: Vite + React + TypeScript (operator & admin UI)
- **Backend**: FastAPI (REST APIs, model serving, optimization services, specs management)
- **Training**: Python pipeline with Optuna hyperparameter studies (XGB, LGB, CatBoost, NN) and stacking

---

A) Repository Map

```
OPTI_PARAM_AI/
├─ backend/
│  └─ app/
│     ├─ .env                        # Backend environment (local dev)
│     ├─ api_schemas.py             # Pydantic request/response models
│     ├─ config.py                  # App settings, constants, env loaders
│     ├─ db.py                      # DB connection helpers (if used)
│     ├─ logging_utils.py           # Structured logging setup
│     ├─ main.py                    # FastAPI app; mounts routers; CORS
│     ├─ model_service.py           # Prediction & model-loading endpoints
│     ├─ optimizer_service.py       # Recommendation/optimization endpoints
│     ├─ specs_service.py           # Specs CRUD / retrieval endpoints
│     ├─ requirements.txt           # Backend python deps
│     └─ requirements.lock.txt      # (Optional) locked versions
│
├─ frontend/
│  ├─ History/                      # UI history/snapshots (optional)
│  └─ optiparam-ui/                 # Vite + React + TS app
│     ├─ public/
│     ├─ src/
│     │  ├─ assets/
│     │  ├─ App.css
│     │  ├─ App.tsx
│     │  ├─ index.css
│     │  └─ main.tsx
│     ├─ .gitignore
│     ├─ eslint.config.js
│     ├─ index.html
│     ├─ package.json
│     ├─ package-lock.json
│     ├─ README.md                  # UI-specific notes (optional)
│     ├─ tsconfig*.json
│     └─ vite.config.ts
│  └─ requirements.txt              # (If present: doc-only; node deps are in package.json)
│
├─ models/                           # Deployed models & artifacts
│  ├─ preprocessor.joblib
│  ├─ fill_defaults.json
│  ├─ cat_mappings.json
│  ├─ lgb_Thickness_Point_A.joblib  # etc. per algorithm & point
│  └─ stack_meta_*.joblib
│
├─ sql/                              # DB schema/migrations/queries (if used)
│
├─ training/
│  ├─ __pycache__/
│  ├─ _lmodels/                      # Legacy model dumps (if any)
│  ├─ catboost_info/                 # CatBoost runtime folder
│  ├─ History/                       # Study logs/checkpoints (optional)
│  ├─ models/                        # Training outputs (intermediate)
│  ├─ models_v3/                     # Another versioned output
│  ├─ trainEnv1/                     # Python venv folder (local training environment)
│  ├─ .env                           # Training-specific env
│  ├─ Data_1.xlsx                    # Sample/raw training data
│  └─ requirements.txt               # Training python deps
│
├─ optiEnv1/                         # Another venv (project environment)
├─ train_xgb.py                      # Standalone training script (legacy)
├─ utils.py                          # Shared helpers used by training
├─ requirements.txt                  # (Repo-level note file if used)
└─ .gitignore
```


---

B) How the pieces fit together

1. **Training** produces model artifacts into `models/`:
   - Per-algorithm models for each thickness point (A–G)
   - `preprocessor.joblib` (feature pipeline for inference)
   - `fill_defaults.json` (safe fill values for missing cols)
   - `cat_mappings.json` (categorical value lists)
   - `stack_meta_*.joblib` (stacking meta-models)

2. **Backend** loads the artifacts from `models/` at startup (see `model_service.py`) and exposes:
   - **Prediction** endpoints (e.g., `/api/predict`) using `preprocessor.joblib` + base models + stacking
   - **Recommendation/Optimization** endpoints (in `optimizer_service.py`)
   - **Specs** endpoints (in `specs_service.py`)

3. **Frontend** calls backend APIs (base URL from `VITE_API_BASE_URL`) to:
   - Show predictions, optimization results, current specs
   - Support admin actions (upload models/reload, if implemented)
   - Provide operator-friendly UI (highlight changed cells, progress, etc.)

---

C) Quickstart (Local Development)

Prereqs
- **Python 3.10+**
- **Node 18+ / 20+** (LTS)
- Optional: **SQL Server** (if you log trials or store specs in MSSQL)

C.1 Backend (FastAPI)
```bash
cd backend/app
python -m venv .venv       # or use your existing venv
. .venv/Scripts/activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
# create .env (see "Backend .env" below)

uvicorn main:app --reload --port 8000
```

**Backend .env (example)**
```
MODEL_DIR=../../models
# CORS allowed origins (comma-separated) e.g. http://localhost:5173
ALLOWED_ORIGINS=*
# DB (if used):
MSSQL_HOST=...
MSSQL_DB=...
MSSQL_USER=...
MSSQL_PASSWORD=...
```

> See `config.py` for recognized env vars. `main.py` usually mounts routers and sets **CORS**.  
> If UI can’t call the API from another host, verify CORS in `main.py`.

C.2 Frontend (Vite + React + TS)
```bash
cd frontend/optiparam-ui
npm ci
# create .env.local with VITE_API_BASE_URL
npm run dev
# Dev server ≈ http://localhost:5173
```

**Frontend `.env.local` (example)**
```
VITE_API_BASE_URL=http://localhost:8000
```

**Expose the UI on your LAN** (optional)  
Update `vite.config.ts`:
```ts
export default defineConfig({
  server: {
    host: true,      // or "0.0.0.0"
    port: 5173,
    strictPort: true
  }
});
```
Then open `http://<your-PC-LAN-IP>:5173` on other devices.

---

D) Training: Optuna Studies + Stacking

Training logic lives in `training/` and **uses** helpers from `utils.py`.  
Your repo also contains legacy scripts like `train_xgb.py`.

D.1 Environments
Create a training venv (or reuse `trainEnv1/`):
```bash
cd training
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

D.2 Data
- Place source data (e.g., `Data_1.xlsx`) or a CSV where your training script expects it.
- `utils.py` should expose:
  - `read_table(path)` — reads CSV/XLSX
  - `add_engineered_features(df)` — feature engineering
  - `build_feature_lists(df)` / `select_features(...)`
  - `build_preprocessor(...)` — training/inference pipeline
  - `THICKNESS_COLS` — columns for A–G
  - `winsorize_targets(...)`, `_collect_specs_from_df(...)`

D.3 Running studies (version with per-algo studies)
Your *current* main script (example name):
- `train_auto_opt_full_fixed.py` (per-algorithm studies)
- It runs **four separate** Optuna studies (`xgb`, `lgb`, `cat`, `nn`) and writes artifacts into `models/`.

**Key env knobs**
```
# Which algos to run
ALGOS=xgb,lgb,cat,nn

# How to schedule studies
# Options vary by script version:
#  - "sequential" or "parallel"
#  - (If enabled in your script) "roundrobin" with rotation:
MULTI_ALGO_MODE=sequential

# Parallelism within a study
OPTUNA_PARALLEL_JOBS=1

# Trials per run (if not round-robin)
OPTUNA_TRIALS=20000

# Round-robin mode (if your script supports it)
# ROUND_TRIALS=5000
# ALGO_ORDER=nn,xgb,cat,lgb
# TARGET_TRIALS_NN=20000
# TARGET_TRIALS_XGB=20000
# TARGET_TRIALS_CAT=20000
# TARGET_TRIALS_LGB=20000
```

**Optuna storage**
```
OPTUNA_STORAGE_TYPE=sqlite         # or mssql
OPTUNA_SQLITE_PATH=./models/optuna_study.db

# If using MSSQL:
MSSQL_HOST=...
MSSQL_DB=...
MSSQL_USER=...
MSSQL_PASSWORD=...
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
```

**Run**
```bash
python train_auto_opt_full_fixed.py
# or set FINAL_ONLY=1 to skip studies and just finalize from saved best trials.
```

**Outputs written to `models/`:**
- `preprocessor.joblib`
- `fill_defaults.json`
- `cat_mappings.json`
- `{algo}_{Point}.joblib` for A–G (e.g., `lgb_Thickness_Point_A.joblib`)
- `stack_meta_{Point}.joblib`
- Optional: `optuna_study_final_*_ALGO.joblib`, `optuna_trials_*.csv`, dashboard CSVs

---

E) Backend Services — where to look

- **`main.py`**  
  Creates the FastAPI app, sets CORS, includes routers. If you need to change base paths or docs URLs, do it here.

- **`api_schemas.py`**  
  Pydantic models for request/response. Add fields here when UI/backend contracts change.

- **`model_service.py`**  
  Loads artifacts from `models/` on startup and exposes **prediction** (and optionally stack) endpoints.  
  Look here if you need to:
  - Change model paths (`MODEL_DIR`)
  - Add a **reload** endpoint after re-training
  - Control batch prediction formats

- **`optimizer_service.py`**  
  Implements optimization/recommendation endpoints.  
  - Where to adjust objective functions, constraints, or “highlight changed cells” payloads for the UI.

- **`specs_service.py`**  
  Reads/writes **specs** used both by training (weighting/normalization) and by operator UI.  
  - If specs are DB-backed, this service calls `db.py`.

- **`config.py`**  
  Centralizes config/env. If you add new env vars, define defaults here.

- **`logging_utils.py`**  
  Standard logging. Use it for consistent timestamps, levels, and rotating logs.

> **Finding routes quickly**: search for `APIRouter(` and `@router` in `*_service.py` files.

---

F) Frontend — where to look

- **`src/App.tsx`**  
  Root UI with route/layout. Hooks into API services and state.  
  If you need new screens or tabs, start here.

- **`src/main.tsx`**  
  React bootstrap. Vite entrypoint.

- **`src/index.css` / `src/App.css`**  
  Global styles + component styles (including table row color rules).  
  - For sticky top bars: check `position: sticky`/`fixed`.
  - For table color contrast issues: ensure text color overrides handle both light/dark row backgrounds.

- **`vite.config.ts`**  
  Dev server, proxy rules, HMR on LAN.  
  Add `server.proxy` if you prefer `/api` to be proxied to `http://localhost:8000`.

---

G) Updating Models in Production

1. Run training → verify artifacts in `models/`.
2. Restart backend or call a **/reload** endpoint (if implemented) so `model_service` re-loads from disk.
3. Smoke test:
   - `GET /health` (if present)
   - Try a small `POST /predict` payload from the UI or cURL.

---

H) SQL Folder

- Put DDL (create tables), DML (seed), and ad-hoc queries here.
- Typical tables:
  - `optuna_trials` (if logging trials to DB)
  - `specs` / `specs_history`
  - Any operational tables (e.g., progress tracking)

> Keep connection strings out of SQL files—use env in `db.py` / `config.py`.


I) Coding Conventions

- **Backend**: Pydantic for schemas, Routers per domain (`*_service.py`), config via env.  
- **Frontend**: TypeScript, functional components, keep API calls in dedicated helpers/hooks.  
- **Training**: Reproducible seeds, per-algo Optuna studies, stacking meta-learner, artifacts saved to `models/`.

---

J) Useful Commands

**Backend**
```bash
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
npm run dev
npm run build
npm run preview
```

**Training**
```bash
# sequential studies
ALGOS=xgb,lgb,cat,nn MULTI_ALGO_MODE=sequential OPTUNA_TRIALS=20000 python train_auto_opt_full_fixed.py

# (if supported) round-robin rotation
MULTI_ALGO_MODE=roundrobin ROUND_TRIALS=5000 ALGO_ORDER=nn,xgb,cat,lgb TARGET_TRIALS_NN=20000 TARGET_TRIALS_XGB=20000 TARGET_TRIALS_CAT=20000 TARGET_TRIALS_LGB=20000 python train_auto_opt_full_fixed.py
```

---

K) Contributing

- Add/adjust endpoints under the relevant `*_service.py`.
- Update API contracts in `api_schemas.py` and UI types accordingly.
- Keep training utilities centralized in `utils.py`.
- Document new env vars in this README.

