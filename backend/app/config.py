import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    here = Path(__file__).resolve()
    candidates = [here.with_name(".env"), here.parent.parent / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p)
            break
except Exception:
    pass

MSSQL_DRIVER  = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
MSSQL_HOST    = os.getenv("MSSQL_HOST", "")
MSSQL_DB      = os.getenv("MSSQL_DB", "")
MSSQL_USER    = os.getenv("MSSQL_USER", "")
MSSQL_PASS    = os.getenv("MSSQL_PASSWORD", "")

# Resolve relative paths against backend/app/
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = os.getenv("MODEL_DIR", str(ROOT_DIR / "models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
