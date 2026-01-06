import os
import joblib
import optuna
from sqlalchemy import create_engine

# Same storage you used during training
STORAGE = "mssql+pyodbc://olj1sgh:rbsz2025@(localdb)\\MSSQLLocalDB/OptiParamAI?driver=ODBC+Driver+17+for+SQL+Server"

STUDY_NAME = "opti_v1.1"
MODEL_DIR = "..\models"
MODEL_VERSION = "v1.5"   # same version you want to save under

os.makedirs(MODEL_DIR, exist_ok=True)

def export_final_study():
    print(f"Connecting to Optuna storage: {STORAGE}")
    
    # Load study DIRECTLY from database
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    final_path = os.path.join(MODEL_DIR, f"optuna_study_final_{MODEL_VERSION}.joblib")
    print(f"Saving final study to: {final_path}")

    joblib.dump(study, final_path)

    print("\n✅ Final Optuna study exported successfully!")
    print(f"✅ File created: {final_path}")
    print(f"✅ Trials in DB: {len(study.trials)}")
    print(f"✅ Best trial value: {study.best_value}")
    print(f"✅ Best params: {study.best_params}")


if __name__ == "__main__":
    export_final_study()
