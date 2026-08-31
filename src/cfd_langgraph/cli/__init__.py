from __future__ import annotations

from typing import Any, List, Optional

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> Any:
    """Lazy re-export of ``cli.repl.main``.

    Deliberately not a module-level ``from .repl import main``: importing this
    package must stay cheap and side-effect-free, because ``manager/tools.py``
    imports ``cfd_langgraph.cli.activity`` to report tool progress. An eager
    import here would drag repl -> manager -> tools back through a
    partially-initialised module and make that a circular import.
    """
    from .repl import main as _main

    return _main(argv)
