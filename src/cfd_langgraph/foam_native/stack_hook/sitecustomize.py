"""Loaded by every Python process a case's Allrun starts: the case loop puts
this folder on PYTHONPATH (cfd_langgraph.foam_native.loop.run_script). On
SIGWINCH, which nothing else in a run uses and which a process without this
hook ignores, the process writes its Python stack to
$CFD_SCIENTIST_STACK_DIR/stack.<pid> and carries on. The loop asks for these
before it stops a stuck run, so the review is told where each rank waited:
DAFoam scripts deadlocked between ranks four times, and "nothing changed for
45 min" alone did not say where (shelf_demo_dafoam_20261019_1057).

Any sitecustomize further down the path (Ubuntu's own) is loaded after it."""

import os

_dir = os.environ.get("CFD_SCIENTIST_STACK_DIR")
if _dir and os.path.isdir(_dir):
    try:
        import faulthandler
        import signal

        _file = open(os.path.join(_dir, f"stack.{os.getpid()}"), "w")
        faulthandler.register(signal.SIGWINCH, file=_file, all_threads=True)
    except Exception:
        pass

try:
    import importlib.machinery
    import importlib.util
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.machinery.PathFinder.find_spec(
        "sitecustomize", [p for p in sys.path if os.path.abspath(p or ".") != _here])
    if _spec and _spec.loader:
        _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
except Exception:
    pass
