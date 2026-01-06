
from sqlalchemy import text
from .db import ENGINE
from .logging_utils import get_logger

log = get_logger("specs_service")


def normalize_type_code(tc: str) -> str:
    return (tc or "").strip().upper()


def ensure_product_type_uniqueness():
    """
    Optional hardening:
    - Adds computed persisted column type_code_norm = UPPER(LTRIM(RTRIM(type_code)))
    - Adds UNIQUE index on type_code_norm

    Safe to call on startup; if permissions are missing, we log and continue.
    """
    try:
        with ENGINE.begin() as c:
            # 1) add computed column if missing
            col_exists = c.execute(
                text("SELECT COL_LENGTH('dbo.product_type', 'type_code_norm') AS len")
            ).scalar()

            if col_exists is None:
                log.info("Adding computed column dbo.product_type.type_code_norm ...")
                c.execute(text("""
                    ALTER TABLE dbo.product_type
                    ADD type_code_norm AS UPPER(LTRIM(RTRIM(type_code))) PERSISTED
                """))

            # 2) unique index if missing
            idx_exists = c.execute(text("""
                SELECT 1
                FROM sys.indexes
                WHERE name = 'UX_product_type_type_code_norm'
                  AND object_id = OBJECT_ID('dbo.product_type')
            """)).first()

            if not idx_exists:
                log.info("Creating UNIQUE INDEX UX_product_type_type_code_norm ...")
                c.execute(text("""
                    CREATE UNIQUE INDEX UX_product_type_type_code_norm
                    ON dbo.product_type(type_code_norm)
                """))

    except Exception:

        log.exception("ensure_product_type_uniqueness failed (ignored)")


def get_types():
    with ENGINE.connect() as c:
        rows = c.execute(
            text("SELECT type_code FROM dbo.product_type ORDER BY type_code")
        ).all()
    return [r[0] for r in rows]


def upsert_type(type_code: str, description: str | None = None) -> tuple[int, bool, str]:
    """
    Race-safe upsert:
    - Normalize code (trim+upper)
    - Lock using UPDLOCK+HOLDLOCK so concurrent inserts can't create duplicates
    - Return (type_id, created, normalized_code)

    If description is provided, we only write it when the existing description is NULL/blank.
    """
    tc = normalize_type_code(type_code)
    if not tc:
        raise ValueError("type_code is required")

    select_sql = text("""
        SELECT TOP 1 type_id, type_code, description
        FROM dbo.product_type WITH (UPDLOCK, HOLDLOCK)
        WHERE UPPER(LTRIM(RTRIM(type_code))) = UPPER(:tc)
    """)

    insert_sql = text("""
        INSERT INTO dbo.product_type(type_code, description)
        OUTPUT INSERTED.type_id
        VALUES (:tc, :desc)
    """)

    update_desc_sql = text("""
        UPDATE dbo.product_type
        SET description = :desc
        WHERE type_id = :type_id
          AND (description IS NULL OR LTRIM(RTRIM(description)) = '')
    """)

    with ENGINE.begin() as c:
        row = c.execute(select_sql, {"tc": tc}).mappings().first()

        if row:
            type_id = int(row["type_id"])
            created = False


            if description is not None and str(description).strip() != "":
                c.execute(update_desc_sql, {"desc": description, "type_id": type_id})


            return type_id, created, normalize_type_code(row["type_code"])


        new_id = c.execute(insert_sql, {"tc": tc, "desc": description}).scalar()
        return int(new_id), True, tc


def get_specs(type_code: str):
    tc = normalize_type_code(type_code)
    sql = """
      SELECT s.point, s.lower_um, s.upper_um, s.default_target_um, s.weight
      FROM dbo.thickness_spec s
      JOIN dbo.product_type t ON t.type_id = s.type_id
      WHERE UPPER(t.type_code) = UPPER(:tc)
    """
    with ENGINE.connect() as c:
        rows = c.execute(text(sql), {"tc": tc}).mappings().all()
    return {
        r["point"]: {
            "lo": float(r["lower_um"]),
            "hi": float(r["upper_um"]),
            "target": float(r["default_target_um"]),
            "w": float(r["weight"]),
        }
        for r in rows
    }


def get_feature_bounds(type_code: str):
    tc = normalize_type_code(type_code)
    sql = """
      SELECT fb.feature, fb.q05, fb.q50, fb.q95, fb.is_tunable
      FROM dbo.feature_bounds fb
      JOIN dbo.product_type t ON t.type_id = fb.type_id
      WHERE UPPER(t.type_code) = UPPER(:tc)
      ORDER BY fb.feature
    """
    with ENGINE.connect() as c:
        rows = c.execute(text(sql), {"tc": tc}).mappings().all()
    return {
        r["feature"]: {
            "lo": float(r["q05"]),
            "mid": float(r["q50"]),
            "hi": float(r["q95"]),
            "tunable": bool(r["is_tunable"]),
        }
        for r in rows
    }
