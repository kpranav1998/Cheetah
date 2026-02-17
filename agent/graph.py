"""LangGraph ReAct agent graph with MCP tool integration."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from utils.logger import get_logger, log_node_transition

logger = get_logger("agent.graph")

SYSTEM_PROMPT = """You are a trading analysis assistant. You have access to tools for:
- Fetching and loading market price data (OHLCV)
- Scanning for chart patterns (double bottom/top, head & shoulders, flags, triangles, cup & handle)
- Finding support and resistance levels
- Computing technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP)
- Backtesting trading strategies with full performance metrics
- Running parameter sweeps to find optimal strategy parameters
- Managing live positions and orders via Zerodha Kite

When analyzing data:
1. First fetch or load the data
2. Then apply the relevant analysis tool(s)
3. Present results clearly with key insights

For Indian NSE stocks, use the .NS suffix (e.g. RELIANCE.NS, TCS.NS, INFY.NS).
For Nifty 50 index, use ^NSEI.

Be concise and focus on actionable insights."""


def build_graph(
    tools: list[dict[str, Any]],
    call_model: Any,
    call_tool: Any,
) -> StateGraph:
    """Build the ReAct agent graph.

    Args:
        tools: List of tool schemas (name, description, inputSchema).
        call_model: Async callable(state) -> state with new AIMessage.
        call_tool: Async callable(state) -> state with ToolMessages.
    """

    async def agent_node(state: AgentState) -> dict:
        log_node_transition("agent", f"messages={len(state['messages'])}")
        result = await call_model(state)
        return result

    async def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        log_node_transition("tools", f"tool_calls={len(tool_calls)}")
        result = await call_tool(state)
        return result

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph
