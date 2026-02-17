from __future__ import annotations

import json

from analysis.llm_client import LLMClient
from backtesting.result import BacktestResult


class BacktestResultAnalyzer:
    """Feed backtest results to LLM for narrative analysis."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def analyze(self, result: BacktestResult) -> str:
        system_prompt = (
            "You are an expert quantitative analyst specializing in Indian stock markets (NSE/BSE). "
            "Analyze the following backtest results and provide actionable insights."
        )

        prompt = f"""Analyze this backtest result:

{result.summary_for_llm()}

Provide:
1. Overall assessment of strategy viability
2. Key strengths and weaknesses
3. Market conditions where this strategy performs best/worst
4. Risk-adjusted return analysis (is the Sharpe ratio adequate?)
5. Specific suggestions for parameter tuning or improvement
6. Whether this strategy is suitable for live trading
"""
        return self.llm.query(prompt, system_prompt)

    def compare_strategies(self, results: list[BacktestResult]) -> str:
        system_prompt = (
            "You are an expert quantitative analyst. Compare these backtest results "
            "and recommend the best strategy."
        )

        summaries = []
        for r in results:
            summaries.append(
                f"--- {r.strategy_name} (params={json.dumps(r.params)}) ---\n"
                f"Return: {r.metrics.total_return_pct:.2f}%, "
                f"Sharpe: {r.metrics.sharpe_ratio:.2f}, "
                f"Max DD: {r.metrics.max_drawdown_pct:.2f}%, "
                f"Win Rate: {r.metrics.win_rate_pct:.2f}%, "
                f"Trades: {r.metrics.total_trades}"
            )

        prompt = f"""Compare these strategy backtests on {results[0].symbol}:

{chr(10).join(summaries)}

Which strategy would you recommend and why? Consider risk-adjusted returns,
consistency, and robustness. Provide a final recommendation.
"""
        return self.llm.query(prompt, system_prompt)

    def suggest_parameters(
        self, strategy_name: str, sweep_results: list[BacktestResult]
    ) -> str:
        system_prompt = (
            "You are a quantitative analyst. Analyze parameter sweep results and "
            "suggest optimal parameters, considering overfitting risk."
        )

        rows = []
        for r in sweep_results[:20]:  # Top 20
            rows.append(
                f"Params={json.dumps(r.params)}: "
                f"Return={r.metrics.total_return_pct:.2f}%, "
                f"Sharpe={r.metrics.sharpe_ratio:.2f}, "
                f"MaxDD={r.metrics.max_drawdown_pct:.2f}%, "
                f"Trades={r.metrics.total_trades}"
            )

        prompt = f"""Parameter sweep results for {strategy_name} (sorted by Sharpe ratio):

{chr(10).join(rows)}

1. Which parameter set is optimal and why?
2. Are there signs of overfitting in the top results?
3. What is a robust parameter set that would work across market conditions?
"""
        return self.llm.query(prompt, system_prompt)
