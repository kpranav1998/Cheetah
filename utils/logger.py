from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

# Correlation ID for tracing a single conversation turn / request
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = _request_id.get()
        if rid:
            entry["request_id"] = rid
        if hasattr(record, "duration_ms"):
            entry["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_data"):
            entry["extra"] = record.extra_data
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Text Formatter
# ---------------------------------------------------------------------------

class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        base = f"{ts} [{record.name}] {record.levelname}: {record.getMessage()}"
        parts: list[str] = []
        rid = _request_id.get()
        if rid:
            parts.append(f"request_id={rid}")
        if hasattr(record, "duration_ms"):
            parts.append(f"duration_ms={record.duration_ms:.1f}")
        if hasattr(record, "extra_data") and record.extra_data:
            for k, v in record.extra_data.items():
                parts.append(f"{k}={v}")
        if parts:
            base += " | " + " ".join(parts)
        if record.exc_info and record.exc_info[0] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ---------------------------------------------------------------------------
# Setup (called once)
# ---------------------------------------------------------------------------

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "text").lower()
    log_file = os.getenv("LOG_FILE", "")

    level = getattr(logging, level_str, logging.INFO)

    formatter: logging.Formatter
    if fmt == "json":
        formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = TextFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (rotating)
    if not log_file:
        project_root = Path(__file__).resolve().parent.parent
        log_dir = project_root / "storage" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "trading_agent.log")

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "urllib3", "yfinance", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# get_logger (backwards-compatible API)
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    setup_logging()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Decorators / Helpers
# ---------------------------------------------------------------------------

def log_tool_call(func):
    """Decorator for MCP tool functions — logs invocation, duration, result/error."""
    logger = logging.getLogger(f"mcp.tool.{func.__name__}")

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(
            "Tool invoked",
            extra={"extra_data": {"tool": func.__name__, "params": _safe_params(kwargs)}},
        )
        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = (time.perf_counter() - t0) * 1000
            logger.info(
                "Tool completed",
                extra={"duration_ms": duration, "extra_data": {"tool": func.__name__}},
            )
            return result
        except Exception:
            duration = (time.perf_counter() - t0) * 1000
            logger.exception(
                "Tool failed",
                extra={"duration_ms": duration, "extra_data": {"tool": func.__name__}},
            )
            raise

    return wrapper


def log_node_transition(node_name: str, state_summary: str) -> None:
    logger = logging.getLogger("agent.graph")
    logger.info(
        f"Node: {node_name}",
        extra={"extra_data": {"node": node_name, "state": state_summary}},
    )


def _safe_params(params: dict) -> dict:
    """Truncate large param values for logging."""
    safe: dict[str, Any] = {}
    for k, v in params.items():
        s = str(v)
        safe[k] = s if len(s) <= 200 else s[:200] + "..."
    return safe
