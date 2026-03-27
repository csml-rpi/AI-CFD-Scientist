from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


class PromptLoader:
    def __init__(self, prompts_path: Path):
        self.prompts_path = prompts_path
        self._data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts_path}")
        self._data = yaml.safe_load(self.prompts_path.read_text(encoding="utf-8")) or {}

    def section(self, name: str) -> Dict[str, Any]:
        return self._data.get(name, {})
