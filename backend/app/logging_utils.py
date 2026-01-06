import logging
import os
import sys
from contextvars import ContextVar


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
run_id_ctx: ContextVar[str] = ContextVar("run_id", default="-")

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:

        record.request_id = request_id_ctx.get()
        record.run_id = run_id_ctx.get()
        return True

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "run_id": getattr(record, "run_id", "-"),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def setup_logging(level: str | None = None, fmt: str | None = None):
    """
    Configure root logger with a context filter so every log line includes
    request_id and run_id. Also route uvicorn logs through the same handlers.
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.getenv("LOG_FORMAT", "text")  # "json" or "text"

    root = logging.getLogger()
    if root.handlers:

        return

    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    if fmt.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] req=%(request_id)s run=%(run_id)s %(message)s"
        )
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    root.addHandler(handler)


    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)