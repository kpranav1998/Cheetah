from __future__ import annotations

from analysis.llm_client import LLMClient
from analysis.result_analyzer import BacktestResultAnalyzer
from backtesting.result import BacktestResult
from backtesting.runner import BacktestRunner


class ParameterOptimizer:
    """Run parameter sweep + LLM analysis to find optimal parameters."""

    def __init__(self, runner: BacktestRunner | None = None, llm: LLMClient | None = None):
        self.runner = runner or BacktestRunner()
        self.analyzer = BacktestResultAnalyzer(llm)

    def optimize(
        self,
        strategy_name: str,
        symbol: str,
        timeframe,
        start,
        end,
    ) -> tuple[list[BacktestResult], str]:
        """Run sweep and return (sorted results, LLM analysis)."""
        results = self.runner.run_parameter_sweep(
            strategy_name, symbol, timeframe, start, end
        )

        if not results:
            return [], "No valid parameter combinations found."

        analysis = self.analyzer.suggest_parameters(strategy_name, results)
        return results, analysis
