from __future__ import annotations

import os
import sys
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Any


DEFAULT_SUBPROCESS_TIMEOUT = 1800  # 30 minutes
DEFAULT_STD_TAIL_CHARS = 1000  # keep only tail for JSON payload


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

        stdout_tail = deque(maxlen=DEFAULT_STD_TAIL_CHARS)
        stderr_tail = deque(maxlen=DEFAULT_STD_TAIL_CHARS)

        def _pump(pipe, sink, tail_buf):
            try:
                for line in iter(pipe.readline, ""):
                    sink.write(line)
                    sink.flush()
                    tail_buf.extend(line)
            finally:
                pipe.close()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, stdout_tail), daemon=True)
            t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, stderr_tail), daemon=True)
            t_out.start()
            t_err.start()

            try:
                returncode = proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                t_out.join(timeout=1)
                t_err.join(timeout=1)
                return {
                    "planned": False,
                    "cmd": cmd,
                    "returncode": -1,
                    "stdout": "".join(stdout_tail),
                    "stderr": f"Process timed out after {self.timeout}s\n" + "".join(stderr_tail),
                    "stdout_truncated": True,
                    "stderr_truncated": True,
                }

            t_out.join(timeout=1)
            t_err.join(timeout=1)
            return {
                "planned": False,
                "cmd": cmd,
                "returncode": returncode,
                "stdout": "".join(stdout_tail),
                "stderr": "".join(stderr_tail),
                "stdout_truncated": True,
                "stderr_truncated": True,
            }
        except Exception as e:
            return {
                "planned": False,
                "cmd": cmd,
                "returncode": -1,
                "stdout": "".join(stdout_tail),
                "stderr": f"Runner exception: {e}\n" + "".join(stderr_tail),
                "stdout_truncated": True,
                "stderr_truncated": True,
            }
