"""Entry point for the Trading Agent chat interface.

Usage:
    python run_agent.py                    # Uses default model from settings
    python run_agent.py --model gpt-4o     # Override model
    python run_agent.py --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Trading Agent - AI-powered trading assistant")
    parser.add_argument("--model", type=str, default=None, help="LiteLLM model name (e.g. gpt-4o, claude-sonnet-4-20250514)")
    args = parser.parse_args()

    from agent.chat import main as chat_main
    asyncio.run(chat_main(model=args.model))


if __name__ == "__main__":
    main()
