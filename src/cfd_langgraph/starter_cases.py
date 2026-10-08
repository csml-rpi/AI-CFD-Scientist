"""Resolve the OpenFOAM case directories a starter folder ships.

``starter_understanding.json`` records them under ``base_case_path``, which may
be a list, a single path, or one string naming several. Three things depend on
it: the case context handed to requirement generation, the mesh gate's
``seed_only`` shortcut, and the function-object block generated cases inherit.

Resolution is shared rather than repeated per call site, and distinguishes
"no case was named" from "a case was named and does not resolve" -- the second
is a configuration error, not an absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# A directory is an OpenFOAM case when it can tell a solver what to run.
_CASE_MARKER = ("system", "controlDict")

# How deep to look when the declared value resolves to nothing. Cases live at
# the top of a starter or one level in; deeper than this and we are guessing.
_SCAN_DEPTH = 3

# Separators a model reaches for when asked for "a path" and it has several.
_SEPARATORS = (";", "\n", ",")


def _is_case(path: Path) -> bool:
    return path.joinpath(*_CASE_MARKER).is_file()


def _split_candidates(raw: str) -> List[str]:
    """Split one string into the paths it names."""
    parts = [raw]
    for sep in _SEPARATORS:
        if any(sep in p for p in parts):
            parts = [piece for p in parts for piece in p.split(sep)]
    # Trailing slashes only: stripping both ends turns "/abs/path" into a
    # relative one, which then resolves against the starter and misses.
    cleaned = [p.strip().rstrip("/") for p in parts]
    return [p for p in cleaned if p]


def _resolve_one(candidate: str, starter_dir: Optional[Path]) -> Optional[Path]:
    """A candidate may be absolute, or relative to the starter folder."""
    candidate = candidate.strip().rstrip("/")
    if not candidate:
        return None
    for base in (None, starter_dir):
        if base is None:
            path = Path(candidate)
            if not path.is_absolute():
                continue
        else:
            path = base / candidate
        if _is_case(path):
            return path
    return None


def _resolve_value(raw: str, starter_dir: Optional[Path]) -> List[Path]:
    """The cases one declared value names.

    The whole value is tried as a single path FIRST, so a case directory whose
    own name contains a comma is not torn in half. Only a value that does not
    resolve on its own gets split.
    """
    whole = _resolve_one(raw, starter_dir)
    if whole is not None:
        return [whole]
    found: List[Path] = []
    for candidate in _split_candidates(raw):
        resolved = _resolve_one(candidate, starter_dir)
        if resolved is not None:
            found.append(resolved)
    return found


def _scan(starter_dir: Optional[Path]) -> List[Path]:
    """Every case under the starter, in a stable order."""
    if starter_dir is None or not starter_dir.is_dir():
        return []
    found: List[Path] = []
    for path in sorted(starter_dir.rglob("*")):
        if not path.is_dir():
            continue
        if len(path.relative_to(starter_dir).parts) > _SCAN_DEPTH:
            continue
        if _is_case(path):
            found.append(path)
    return found


@dataclass(frozen=True)
class StarterBaseCases:
    """What the starter says it ships, and what is actually on disk."""

    declared: str
    """Whatever ``base_case_path`` held, rendered for error messages."""

    dirs: List[Path]
    """Resolved case directories, declaration order first, then scan order."""

    scanned: bool
    """True when these came from scanning the starter, not from the declaration."""

    starter_dir: Optional[Path]

    @property
    def primary(self) -> Optional[Path]:
        return self.dirs[0] if self.dirs else None

    @property
    def unresolved(self) -> bool:
        """A case was named, and nothing on disk answers to it."""
        return bool(self.declared) and not self.dirs

    def problem(self) -> str:
        """Empty when there is nothing to report; otherwise say what is wrong.

        Returned rather than raised so a caller that can carry on without a
        seed still can -- but nothing gets to fail quietly.
        """
        if self.unresolved:
            return (
                f"starter_understanding.json names base_case_path "
                f"{self.declared!r}, but no directory under "
                f"{self.starter_dir} with {'/'.join(_CASE_MARKER)} matches it, "
                f"and scanning the starter found no OpenFOAM case either."
            )
        if self.scanned and self.declared:
            return (
                f"starter_understanding.json names base_case_path "
                f"{self.declared!r}, which does not resolve; using the "
                f"{len(self.dirs)} case(s) found by scanning {self.starter_dir} "
                f"instead: {', '.join(d.name for d in self.dirs)}."
            )
        if self.scanned:
            return (
                f"starter_understanding.json names no base_case_path, but "
                f"{self.starter_dir} contains {len(self.dirs)} OpenFOAM case(s): "
                f"{', '.join(d.name for d in self.dirs)}. Using them rather than "
                "treating the starter as having no case."
            )
        return ""


def resolve_starter_base_cases(understanding: Dict[str, Any]) -> StarterBaseCases:
    """Every OpenFOAM case a ``starter_understanding.json`` points at.

    ``base_case_path`` may be a list, a single path, or one string naming
    several paths. A starter with two cases is normal -- one per Reynolds
    number, say -- so this returns all of them rather than forcing a choice
    the data does not support.
    """
    understanding = understanding or {}
    starter_raw = str(understanding.get("starter_dir") or "").strip()
    starter_dir = Path(starter_raw) if starter_raw else None

    value = understanding.get("base_case_path")
    if isinstance(value, (list, tuple)):
        values = [str(item or "").strip() for item in value]
        values = [v for v in values if v]
        declared = "; ".join(values)
    else:
        declared = str(value or "").strip()
        values = [declared] if declared else []

    dirs: List[Path] = []
    for raw in values:
        for resolved in _resolve_value(raw, starter_dir):
            if resolved not in dirs:
                dirs.append(resolved)

    # A declared value that resolves to nothing is the failure this module
    # exists for. Scanning recovers the common case (the model named the
    # cases in a form that is not a path) while `scanned` keeps it visible.
    scanned = False
    if not dirs:
        found = _scan(starter_dir)
        if found:
            dirs = found
            scanned = True

    return StarterBaseCases(
        declared=declared, dirs=dirs, scanned=scanned, starter_dir=starter_dir
    )


def seed_dir_tag(seed: Optional[Path]) -> str:
    """A filename-safe prefix naming the case a copy was seeded from.

    A derived directory (a mesh-gate level) is a copy under a new name, and the
    starter's own name is often what identifies the condition that scoring
    depends on. Returns "" when there is no seed.
    """
    if seed is None:
        return ""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in seed.name)
    safe = safe.strip("_.")[:64].strip("_.")
    return f"{safe}__" if safe else ""


def identity_tokens(*names: str) -> set:
    """The words a name is made of, lowercased, punctuation treated as a gap."""
    out = set()
    for name in names:
        if not name:
            continue
        cleaned = "".join(c if c.isalnum() else " " for c in str(name)).lower()
        out.update(w for w in cleaned.split() if w)
    return out


def match_reference_to_case(case_names: Sequence[str], candidates: Sequence[Path]) -> Optional[Path]:
    """Which of several reference files belongs to this case, if one clearly does.

    A study with several conditions declares one reference file per condition,
    so scoring a case against all of them gives different answers by design.

    Only tokens that distinguish the candidates from each other are counted;
    tokens common to all of them carry no information. Returns None unless
    exactly one candidate wins, leaving an unclear match to the caller.
    """
    files = [Path(c) for c in candidates]
    if len(files) < 2:
        return files[0] if files else None
    per_file = [identity_tokens(f.name) for f in files]
    shared = set.intersection(*per_file) if per_file else set()
    case_tokens = identity_tokens(*case_names)
    scores = [len(case_tokens & (tokens - shared)) for tokens in per_file]
    best = max(scores)
    if best == 0 or scores.count(best) != 1:
        return None
    return files[scores.index(best)]


def starter_base_case_dirs(understanding: Dict[str, Any]) -> List[Path]:
    return resolve_starter_base_cases(understanding).dirs


def starter_base_case_dir(understanding: Dict[str, Any]) -> Optional[Path]:
    return resolve_starter_base_cases(understanding).primary
