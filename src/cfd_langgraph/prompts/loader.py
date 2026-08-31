from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


class PromptLoader:
    def __init__(self, prompts_path: Path, overlay_path: Path | None = None):
        self.prompts_path = prompts_path
        # Optional knowledge_bundle/active_prompts.yaml overlay: promoted,
        # audit-gated prompt tweaks from PromptEvolver. Never required, never
        # mutates prompts_path — deleting the overlay always restores the
        # authoritative prompts.yaml behavior.
        self.overlay_path = overlay_path
        self._data: Dict[str, Any] = {}
        self._overlay: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts_path}")
        self._data = yaml.safe_load(self.prompts_path.read_text(encoding="utf-8")) or {}
        self._overlay = {}
        if self.overlay_path and self.overlay_path.exists():
            self._overlay = yaml.safe_load(self.overlay_path.read_text(encoding="utf-8")) or {}

    def section(self, name: str) -> Dict[str, Any]:
        merged = dict(self._data.get(name, {}))
        merged.update(self._overlay.get(name, {}))
        return merged
