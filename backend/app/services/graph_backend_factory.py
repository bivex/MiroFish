"""Lazy factories for graph-backend-dependent services."""

from __future__ import annotations

from typing import Any

from ..config import Config


SUPPORTED_GRAPH_BACKENDS = {"zep", "cognee"}


def resolve_graph_backend(graph_backend: str | None = None) -> str:
    backend = (graph_backend or Config.get_graph_backend() or "zep").strip().lower()
    if backend not in SUPPORTED_GRAPH_BACKENDS:
        raise ValueError(f"Unsupported GRAPH_BACKEND: {backend}")
    return backend


def validate_graph_backend_requirements(graph_backend: str | None = None) -> list[str]:
    backend = resolve_graph_backend(graph_backend)
    errors: list[str] = []
    if backend == "zep" and not Config.ZEP_API_KEY:
        errors.append("ZEP_API_KEY is not configured")
    return errors


def get_graph_builder_service(graph_backend: str | None = None, **kwargs: Any) -> Any:
    backend = resolve_graph_backend(graph_backend)
    if backend == "zep":
        from .graph_builder import GraphBuilderService

        return GraphBuilderService(**kwargs)

    if backend == "cognee":
        from .cognee_graph_builder import CogneeGraphBuilderService

        return CogneeGraphBuilderService(**kwargs)

    raise NotImplementedError("Unsupported graph builder backend")


def get_entity_reader_service(graph_backend: str | None = None, **kwargs: Any) -> Any:
    backend = resolve_graph_backend(graph_backend)
    if backend == "zep":
        from .zep_entity_reader import ZepEntityReader

        return ZepEntityReader(**kwargs)

    if backend == "cognee":
        from .cognee_entity_reader import CogneeEntityReader

        return CogneeEntityReader(**kwargs)

    raise NotImplementedError("Unsupported entity reader backend")


def get_report_tools_service(graph_backend: str | None = None, **kwargs: Any) -> Any:
    backend = resolve_graph_backend(graph_backend)
    if backend == "zep":
        from .zep_tools import ZepToolsService

        return ZepToolsService(**kwargs)

    if backend == "cognee":
        from .cognee_tools import CogneeToolsService

        return CogneeToolsService(**kwargs)

    raise NotImplementedError("Unsupported report/search backend")


def get_graph_memory_manager_class(graph_backend: str | None = None) -> type[Any]:
    backend = resolve_graph_backend(graph_backend)
    if backend == "zep":
        from .zep_graph_memory_updater import ZepGraphMemoryManager

        return ZepGraphMemoryManager

    raise NotImplementedError("Cognee graph memory updater is not integrated yet")