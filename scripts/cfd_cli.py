#!/usr/bin/env python3
"""Thin entrypoint for the interactive manager CLI, for pip-only installs
(no ``pip install -e .``): ``python scripts/cfd_cli.py run --topic ... --out-dir ...``

Equivalent to the ``cfd-scientist-cli`` console script installed by
``pip install -e .`` (see pyproject.toml). See
``src/cfd_langgraph/cli/repl.py`` for the actual implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfd_langgraph.cli.repl import main  # noqa: E402

if __name__ == "__main__":
    main()
