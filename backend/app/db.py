from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import urllib.parse
import pyodbc

from .config import MSSQL_DRIVER, MSSQL_USER, MSSQL_PASS, MSSQL_HOST, MSSQL_DB
from .logging_utils import get_logger

log = get_logger("db")

def engine():
    driver = (MSSQL_DRIVER or "").strip()
    if not driver:
        preferred = ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server", "SQL Server"]
        available = list(pyodbc.drivers())
        for d in preferred:
            if d in available:
                driver = d
                break
        if not driver:
            raise RuntimeError(f"No suitable ODBC driver found. Available drivers: {available}")

    host = (MSSQL_HOST or "").strip()
    is_localdb = host.lower().startswith("(localdb)")

    if is_localdb:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host};"
            f"DATABASE={MSSQL_DB};"
            f"Trusted_Connection=Yes;"
            f"TrustServerCertificate=Yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host or 'localhost'};"
            f"DATABASE={MSSQL_DB};"
            f"UID={MSSQL_USER};PWD={MSSQL_PASS};"
            f"TrustServerCertificate=Yes;"
        )

    url = URL.create("mssql+pyodbc", query={"odbc_connect": urllib.parse.quote_plus(conn_str)})

    log.info("db.connect driver=%s host=%s db=%s localdb=%s", driver, host, MSSQL_DB, is_localdb)
    return create_engine(url, pool_pre_ping=True, fast_executemany=True)

ENGINE = engine()