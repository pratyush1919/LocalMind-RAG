"""
agents package for LocalMind-RAG.
"""

from agents.browser_agent import (
    WebEvidence,
    WebSearchProvider,
    DuckDuckGoSearchProvider,
    research,
)

__all__ = [
    "WebEvidence",
    "WebSearchProvider",
    "DuckDuckGoSearchProvider",
    "research",
]
