from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_agent.graph import build_router_graph


def main() -> None:
    output_dir = PROJECT_ROOT / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "router_agent_graph.png"

    graph = build_router_graph()
    png_bytes = graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_bytes)
    print(f"Saved router graph image to: {output_path}")


if __name__ == "__main__":
    main()
