"""
graph_store/workflow_dag.py  —  P5-T3
Export the LangGraph agent workflow as a Mermaid DAG and log per-query execution paths.

Concepts covered:
  - Workflow DAGs        (LangGraph graph exported as a visual DAG)
  - Workflow Visualization (Mermaid diagram rendered in Gradio UI + README)

Two outputs:
  1. Static DAG:  full agent workflow as Mermaid → /docs/workflow_dag.md
                  Shown in Gradio "Workflow" tab (P9) and README.

  2. Dynamic DAG: per-query execution path — which nodes actually fired,
                  in what order, with timing → /logs/execution_paths.jsonl
                  Makes agent routing decisions transparent and inspectable.

Why this matters for interviews:
  Generating the DAG from graph.get_graph().draw_mermaid() shows you know
  LangGraph's introspection API. Logging per-query execution paths shows
  you understand that different queries take different routes through the
  graph — multi_hop fires 7 nodes; simple_factual fires only 4.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DAG_PATH = Path("./docs/workflow_dag.md")
_EXEC_LOG_PATH = Path("./logs/execution_paths.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Static DAG export
# ─────────────────────────────────────────────────────────────────────────────

def export_workflow_dag(compiled_graph, force: bool = False) -> str:
    """
    Export the compiled LangGraph as a Mermaid flowchart.

    Saves to /docs/workflow_dag.md.
    Returns the Mermaid string for embedding in the Gradio UI.

    Args:
        compiled_graph: The compiled LangGraph from agents/router.py build_graph().
        force:          If True, re-export even if file already exists.

    Concept: Workflow DAGs
    """
    _DAG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _DAG_PATH.exists() and not force:
        logger.debug("[DAG] Returning cached Mermaid DAG")
        return _DAG_PATH.read_text()

    try:
        mermaid = compiled_graph.get_graph().draw_mermaid()
        content = f"# LocalMind — Agent Workflow DAG\n\n```mermaid\n{mermaid}\n```\n"
        _DAG_PATH.write_text(content)
        logger.info(f"[DAG] Workflow DAG saved to {_DAG_PATH}")
        return content
    except Exception as e:
        logger.warning(f"[DAG] Failed to export Mermaid DAG: {e}")
        # Return a hand-crafted fallback DAG
        return _fallback_mermaid_dag()


def _fallback_mermaid_dag() -> str:
    """
    Hand-crafted Mermaid DAG used when LangGraph introspection fails
    (e.g. when graph is not yet compiled).
    Accurately represents the full 7-agent LocalMind graph.
    """
    return """# LocalMind — Agent Workflow DAG

```mermaid
flowchart TD
    START([__start__]) --> supervisor

    supervisor -->|multi_hop / comparative| planner
    supervisor -->|simple_factual / web_needed| researcher
    supervisor -->|conversational| generator

    planner --> researcher

    researcher --> generator
    generator --> critic
    critic --> reviewer
    reviewer --> router

    router -->|consensus < 7.5 AND iter < 2| researcher
    router -->|consensus >= 7.5 OR iter >= 2| END([__end__])

    style supervisor fill:#e94560,color:#fff
    style planner fill:#0f3460,color:#fff
    style researcher fill:#16213e,color:#fff
    style generator fill:#1a1a2e,color:#fff
    style critic fill:#533483,color:#fff
    style reviewer fill:#05386b,color:#fff
    style router fill:#379683,color:#fff
```
"""


def get_mermaid_dag(compiled_graph=None) -> str:
    """
    Return the Mermaid DAG string for embedding in Gradio or README.
    Uses cached file if available, otherwise exports fresh.
    """
    if _DAG_PATH.exists():
        return _DAG_PATH.read_text()
    if compiled_graph is not None:
        return export_workflow_dag(compiled_graph)
    return _fallback_mermaid_dag()


# ─────────────────────────────────────────────────────────────────────────────
# Per-query execution path logging
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionTracker:
    """
    Tracks which agent nodes fire during a single query execution.

    Usage:
        tracker = ExecutionTracker(run_id, query)
        tracker.record("supervisor", duration_ms=45)
        tracker.record("researcher", duration_ms=320)
        ...
        tracker.save()

    Concept: Workflow DAGs (per-query sub-DAG of actual execution path)
    """

    def __init__(self, run_id: str, query: str):
        self.run_id = run_id
        self.query = query
        self.steps: List[Dict] = []
        self._start = time.time()

    def record(self, node_name: str, duration_ms: Optional[float] = None) -> None:
        """Record a node that fired during this execution."""
        self.steps.append({
            "node": node_name,
            "duration_ms": round(duration_ms, 1) if duration_ms else None,
            "elapsed_ms": round((time.time() - self._start) * 1000, 1),
        })

    def to_dict(self) -> Dict:
        """Serialize the execution path to a loggable dict."""
        nodes_fired = [s["node"] for s in self.steps]
        total_ms = round((time.time() - self._start) * 1000, 1)

        return {
            "run_id": self.run_id,
            "query": self.query[:120],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nodes_fired": nodes_fired,
            "step_count": len(nodes_fired),
            "total_ms": total_ms,
            "steps": self.steps,
            "path_type": _classify_path(nodes_fired),
        }

    def save(self) -> None:
        """Append execution path to JSONL log."""
        _EXEC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_EXEC_LOG_PATH, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
        logger.debug(f"[DAG] Execution path saved for run_id={self.run_id}")

    def to_mermaid(self) -> str:
        """
        Generate a Mermaid flowchart of the actual execution path for this query.
        Highlights which nodes fired (in red) vs which were skipped (greyed).
        """
        all_nodes = ["supervisor", "planner", "researcher", "generator", "critic", "reviewer", "router"]
        fired = set(s["node"] for s in self.steps)

        lines = ["```mermaid", "flowchart LR"]
        for i, step in enumerate(self.steps):
            node = step["node"]
            ms = f"\\n{step['elapsed_ms']}ms" if step.get("elapsed_ms") else ""
            lines.append(f'    {node}["{node}{ms}"]')
            if i > 0:
                prev = self.steps[i - 1]["node"]
                lines.append(f"    {prev} --> {node}")

        # Style fired nodes
        for node in fired:
            lines.append(f"    style {node} fill:#e94560,color:#fff")

        # Grey out skipped nodes
        for node in all_nodes:
            if node not in fired:
                lines.append(f"    style {node} fill:#333,color:#666")

        lines.append("```")
        return "\n".join(lines)


def _classify_path(nodes_fired: List[str]) -> str:
    """
    Classify the execution path type based on which nodes fired.
    Used for analytics and Gradio display.
    """
    if "planner" in nodes_fired and nodes_fired.count("researcher") > 1:
        return "complex_with_refinement"
    elif "planner" in nodes_fired:
        return "complex"
    elif nodes_fired.count("researcher") > 1:
        return "simple_with_refinement"
    elif "generator" in nodes_fired and "researcher" not in nodes_fired:
        return "memory_only"
    else:
        return "simple"


# ─────────────────────────────────────────────────────────────────────────────
# Execution path analytics (for Gradio telemetry tab P9)
# ─────────────────────────────────────────────────────────────────────────────

def get_execution_stats() -> Dict:
    """
    Read the execution_paths.jsonl log and return aggregate statistics.
    Used by the Gradio telemetry tab (P9).
    """
    if not _EXEC_LOG_PATH.exists():
        return {"total_runs": 0, "avg_ms": 0, "path_type_counts": {}}

    runs = []
    with open(_EXEC_LOG_PATH) as f:
        for line in f:
            try:
                runs.append(json.loads(line.strip()))
            except Exception:
                pass

    if not runs:
        return {"total_runs": 0, "avg_ms": 0, "path_type_counts": {}}

    avg_ms = sum(r.get("total_ms", 0) for r in runs) / len(runs)
    path_counts: Dict[str, int] = {}
    for r in runs:
        pt = r.get("path_type", "unknown")
        path_counts[pt] = path_counts.get(pt, 0) + 1

    return {
        "total_runs": len(runs),
        "avg_ms": round(avg_ms, 1),
        "path_type_counts": path_counts,
        "recent_runs": runs[-5:],  # last 5 for display
    }
