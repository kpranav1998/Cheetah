"""Terminal chat loop with MCP tool integration and LangGraph agent."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import litellm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.graph import build_graph, SYSTEM_PROMPT
from agent.state import AgentState
from config.settings import settings
from utils.logger import get_logger, set_request_id

logger = get_logger("agent.chat")
console = Console()


def _mcp_schema_to_openai_tool(tool: dict) -> dict:
    """Convert MCP tool schema to OpenAI-style function tool definition."""
    input_schema = tool.get("inputSchema", {})
    # Ensure we have a valid properties dict
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


async def main(model: str | None = None) -> None:
    """Run the terminal chat loop."""
    model_name = model or settings.litellm_model
    temperature = settings.agent_temperature

    console.print(
        Panel(
            f"[bold green]Trading Agent[/bold green]\n"
            f"Model: [cyan]{model_name}[/cyan] | Temp: {temperature}\n"
            f"Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
            title="Welcome",
            border_style="green",
        )
    )

    # Start MCP server as subprocess
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=_PROJECT_ROOT,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Fetch tools from MCP server
            tools_response = await session.list_tools()
            mcp_tools = [
                {"name": t.name, "description": t.description or "", "inputSchema": t.inputSchema}
                for t in tools_response.tools
            ]
            openai_tools = [_mcp_schema_to_openai_tool(t) for t in mcp_tools]

            logger.info(f"Connected to MCP server with {len(mcp_tools)} tools")
            console.print(f"[dim]Loaded {len(mcp_tools)} tools from MCP server[/dim]\n")

            # Build model caller and tool caller
            async def call_model(state: AgentState) -> dict:
                messages = _to_litellm_messages(state["messages"])
                response = await litellm.acompletion(
                    model=model_name,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    temperature=temperature,
                )
                choice = response.choices[0].message

                # Build AIMessage
                content = choice.content or ""
                tool_calls = []
                if choice.tool_calls:
                    for tc in choice.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments),
                        })

                ai_msg = AIMessage(content=content, tool_calls=tool_calls)
                return {"messages": [ai_msg]}

            async def call_tool(state: AgentState) -> dict:
                last_message = state["messages"][-1]
                tool_messages = []
                for tc in last_message.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    logger.info(f"Calling MCP tool: {tool_name} with {tool_args}")
                    console.print(f"  [dim]Calling tool: {tool_name}[/dim]")

                    try:
                        result = await session.call_tool(tool_name, tool_args)
                        # Extract text content from MCP result
                        if result.content:
                            text_parts = [c.text for c in result.content if hasattr(c, "text")]
                            content = "\n".join(text_parts)
                        else:
                            content = "No result returned."
                    except Exception as e:
                        content = json.dumps({"error": str(e)})
                        logger.error(f"Tool {tool_name} failed: {e}")

                    tool_messages.append(
                        ToolMessage(content=content, tool_call_id=tc["id"])
                    )
                return {"messages": tool_messages}

            # Build and compile graph
            graph = build_graph(mcp_tools, call_model, call_tool)
            compiled = graph.compile()

            # Chat loop
            conversation: list = [SystemMessage(content=SYSTEM_PROMPT)]

            while True:
                try:
                    user_input = console.input("[bold blue]You>[/bold blue] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Goodbye![/dim]")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                # Set correlation ID for this turn
                set_request_id(str(uuid.uuid4())[:8])

                conversation.append(HumanMessage(content=user_input))

                try:
                    state: AgentState = {"messages": list(conversation)}
                    result = await compiled.ainvoke(state)

                    # Extract new messages added by the graph
                    new_messages = result["messages"][len(conversation):]
                    conversation.extend(new_messages)

                    # Display the final assistant response
                    for msg in reversed(new_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            console.print()
                            console.print(Panel(Markdown(msg.content), title="Agent", border_style="cyan"))
                            console.print()
                            break

                except Exception as e:
                    logger.exception("Error during agent invocation")
                    console.print(f"[red]Error: {e}[/red]\n")


def _to_litellm_messages(messages: list) -> list[dict]:
    """Convert LangChain messages to LiteLLM-compatible dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
    return result
