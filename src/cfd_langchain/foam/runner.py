from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Any


DEFAULT_SUBPROCESS_TIMEOUT = 1800  # 30 minutes


class FoamAgentRunner:
    def __init__(self, foam_main: Path, openfoam_path: str, timeout: int = DEFAULT_SUBPROCESS_TIMEOUT):
        self.foam_main = foam_main
        self.openfoam_path = openfoam_path
        self.timeout = timeout

    def plan_command(self, user_requirement_path: Path, output_dir: Path) -> list[str]:
        return [
            "python",
            str(self.foam_main),
            "--openfoam_path",
            self.openfoam_path,
            "--output",
            str(output_dir),
            "--prompt_path",
            str(user_requirement_path),
        ]

    def run(self, user_requirement_path: Path, output_dir: Path, project_root: Path, execute: bool = False) -> Dict[str, Any]:
        cmd = self.plan_command(user_requirement_path, output_dir)
        if not execute:
            return {"planned": True, "cmd": cmd, "cwd": str(project_root)}

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "planned": False,
                "cmd": cmd,
                "returncode": -1,
                "stdout": (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                "stderr": f"Process timed out after {self.timeout}s",
            }
        return {
            "planned": False,
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
