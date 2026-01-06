This folder stores trained artifacts produced by training/train_xgb.py:

- preprocessor.joblib         # sklearn ColumnTransformer (imputers + OneHotEncoder)
- xgb_A.joblib ... xgb_G.joblib
- feature_info.json           # ordered feature list and model version

After running the training script, copy or mount these files here. The backend will
load them on startup and serve predictions and recommendations.