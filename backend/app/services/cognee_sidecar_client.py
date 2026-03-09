"""Client for the isolated Cognee sidecar runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


SENTINEL = "__MIROFISH_COGNEE_JSON__"


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

    def _run(self, operation: str, graph_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.sidecar_python.exists():
            raise FileNotFoundError(f"Cognee sidecar python not found: {self.sidecar_python}")
        if not self.sidecar_script.exists():
            raise FileNotFoundError(f"Cognee sidecar script not found: {self.sidecar_script}")

        workspace = self.ensure_workspace(graph_id)
        env = os.environ.copy()
        env.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

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
                "llm_api_key": llm_api_key,
                "llm_endpoint": llm_endpoint,
                "llm_model": llm_model,
                "llm_provider": "openai",
            },
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        return self._run("get_graph_data", graph_id, {"graph_id": graph_id, "dataset_name": graph_id})

    def search_graph(self, graph_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        return self._run(
            "search_graph",
            graph_id,
            {"graph_id": graph_id, "dataset_name": graph_id, "query": query, "limit": limit},
        )

    def delete_graph(self, graph_id: str) -> None:
        workspace = self.get_workspace_dir(graph_id)
        if workspace.exists():
            shutil.rmtree(workspace)