"""Client for the isolated Cognee sidecar runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List


SENTINEL = "__MIROFISH_COGNEE_JSON__"
_GRAPH_LOCKS: Dict[str, threading.Lock] = {}
_GRAPH_LOCKS_GUARD = threading.Lock()


class CogneeSidecarClient:
    def __init__(self):
        mirofish_root = Path(__file__).resolve().parents[3]
        self.sidecar_root = Path(os.environ.get("COGNEE_SIDECAR_ROOT", mirofish_root / "cognee_sidecar"))
        self.sidecar_python = Path(
            os.environ.get("COGNEE_SIDECAR_PYTHON", self.sidecar_root / ".venv" / "bin" / "python")
        )
        self.sidecar_script = Path(
            os.environ.get("COGNEE_SIDECAR_SCRIPT", self.sidecar_root / "bridge.py")
        )
        self.workspace_root = self.sidecar_root / "workspaces"

    def get_workspace_dir(self, graph_id: str) -> Path:
        return self.workspace_root / graph_id

    def ensure_workspace(self, graph_id: str) -> Path:
        workspace = self.get_workspace_dir(graph_id)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _get_graph_lock(self, graph_id: str) -> threading.Lock:
        with _GRAPH_LOCKS_GUARD:
            lock = _GRAPH_LOCKS.get(graph_id)
            if lock is None:
                lock = threading.Lock()
                _GRAPH_LOCKS[graph_id] = lock
            return lock

    def _run(self, operation: str, graph_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.sidecar_python.exists():
            raise FileNotFoundError(f"Cognee sidecar python not found: {self.sidecar_python}")
        if not self.sidecar_script.exists():
            raise FileNotFoundError(f"Cognee sidecar script not found: {self.sidecar_script}")

        workspace = self.ensure_workspace(graph_id)
        env = os.environ.copy()
        env.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

        with self._get_graph_lock(graph_id):
            result = subprocess.run(
                [str(self.sidecar_python), str(self.sidecar_script), operation],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                cwd=workspace,
                env=env,
                check=False,
            )

        raw_output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        payload_line = None
        for line in reversed(raw_output.splitlines()):
            if line.startswith(SENTINEL):
                payload_line = line[len(SENTINEL) :]
                break

        if payload_line is None:
            raise RuntimeError(raw_output.strip() or f"Cognee sidecar failed with code {result.returncode}")

        parsed = json.loads(payload_line)
        if result.returncode != 0 or not parsed.get("success"):
            raise RuntimeError(parsed.get("error") or raw_output.strip())

        return parsed["data"]

    def _runtime_payload(
        self,
        *,
        llm_api_key: str | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"llm_provider": "openai"}
        resolved_api_key = llm_api_key if llm_api_key is not None else os.environ.get("LLM_API_KEY")
        resolved_endpoint = llm_endpoint if llm_endpoint is not None else os.environ.get("LLM_BASE_URL")
        resolved_model = llm_model if llm_model is not None else (
            os.environ.get("REPORT_LLM_MODEL") or os.environ.get("LLM_MODEL_NAME")
        )

        if resolved_api_key:
            payload["llm_api_key"] = resolved_api_key
        if resolved_endpoint:
            payload["llm_endpoint"] = resolved_endpoint
        if resolved_model:
            payload["llm_model"] = resolved_model

        return payload

    def build_graph(
        self,
        graph_id: str,
        graph_name: str,
        texts: List[str],
        llm_api_key: str | None,
        llm_endpoint: str | None,
        llm_model: str | None,
    ) -> Dict[str, Any]:
        return self._run(
            "build_graph",
            graph_id,
            {
                "graph_id": graph_id,
                "dataset_name": graph_id,
                "graph_name": graph_name,
                "texts": texts,
                **self._runtime_payload(
                    llm_api_key=llm_api_key,
                    llm_endpoint=llm_endpoint,
                    llm_model=llm_model,
                ),
            },
        )

    def get_graph_data(
        self,
        graph_id: str,
        *,
        llm_api_key: str | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
    ) -> Dict[str, Any]:
        return self._run(
            "get_graph_data",
            graph_id,
            {
                "graph_id": graph_id,
                "dataset_name": graph_id,
                **self._runtime_payload(
                    llm_api_key=llm_api_key,
                    llm_endpoint=llm_endpoint,
                    llm_model=llm_model,
                ),
            },
        )

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        *,
        llm_api_key: str | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
    ) -> Dict[str, Any]:
        return self._run(
            "search_graph",
            graph_id,
            {
                "graph_id": graph_id,
                "dataset_name": graph_id,
                "query": query,
                "limit": limit,
                **self._runtime_payload(
                    llm_api_key=llm_api_key,
                    llm_endpoint=llm_endpoint,
                    llm_model=llm_model,
                ),
            },
        )

    def delete_graph(self, graph_id: str) -> None:
        with self._get_graph_lock(graph_id):
            workspace = self.get_workspace_dir(graph_id)
            if workspace.exists():
                shutil.rmtree(workspace)