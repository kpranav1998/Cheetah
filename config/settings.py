from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Paths
    project_root: Path = _PROJECT_ROOT
    cache_dir: Path = _PROJECT_ROOT / "storage" / "cache"
    results_dir: Path = _PROJECT_ROOT / "storage" / "results"
    reports_dir: Path = _PROJECT_ROOT / "storage" / "reports"

    # Kite Connect
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""

    # LLM
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"

    # Agent
    litellm_model: str = "gpt-4o"
    agent_temperature: float = 0.1

    # Logging
    log_level: str = "INFO"
    log_format: str = "text"  # "json" or "text"
    log_file: str = ""  # empty = storage/logs/trading_agent.log

    # Backtesting defaults
    default_capital: float = 1_000_000.0
    default_commission_pct: float = 0.03
    default_slippage_pct: float = 0.01

    def model_post_init(self, __context) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


settings = AppSettings()
