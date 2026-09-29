"""
API Routes package.
"""

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.documents import router as document_router
from api.routes.sessions import router as session_router
from api.routes.graph import router as graph_router

__all__ = [
    "health_router",
    "chat_router",
    "document_router",
    "session_router",
    "graph_router",
]
