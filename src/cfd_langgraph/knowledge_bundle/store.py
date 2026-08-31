from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_root() -> Path:
    return Path(os.environ.get("CFD_KNOWLEDGE_BUNDLE_DIR", str(_REPO_ROOT / "knowledge_bundle")))


class KnowledgeBundle:
    """The thing self-evolution reads from and writes to, across studies.

    There is no pre-seeded corpus. A fresh checkout has an empty
    ``knowledge_bundle/`` — no lessons, no validation cases, no promoted
    skills or prompt variants. Every entry below is earned by a real study
    that finished and passed ``scripts/stage_gate_audit.py``. Call
    :meth:`record_study` exactly once per audited study (see
    ``scripts/audit_and_record.py``), and nowhere else — that is the only
    door into this directory, by design.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else _default_root()
        self.lessons_path = self.root / "lessons.jsonl"
        self.validation_dir = self.root / "validation_suite"
        self.skills_dir = self.root / "skills"
        self.variants_dir = self.root / "prompt_variants"
        self.active_prompts_path = self.root / "active_prompts.yaml"

        for d in (self.root, self.validation_dir, self.skills_dir, self.variants_dir):
            d.mkdir(parents=True, exist_ok=True)

        skills_manifest = self.skills_dir / "manifest.json"
        if not skills_manifest.exists():
            skills_manifest.write_text(json.dumps({"skills": []}, indent=2))
        if not self.active_prompts_path.exists():
            self.active_prompts_path.write_text("{}\n")

    # ---------------------------------------------------------------- record

    def record_study(self, out_dir: Path, *, extra_lessons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Call once, right after ``stage_gate_audit.py`` exits 0 for ``out_dir``.

        Appends a Reflexion-style lessons entry (plain-language notes, no
        prompt/weight change involved) and promotes ``out_dir`` into the
        validation suite that gates every later self-evolution proposal.
        """
        out_dir = Path(out_dir).resolve()
        state = _read_json(out_dir / "state.json") or {}
        audit = _read_json(out_dir / "audit_passed.json") or {}

        lessons = list(extra_lessons or [])
        lessons.extend(self._derive_lessons(out_dir))

        entry = {
            "study_id": out_dir.name,
            "study_dir": str(out_dir),
            "recorded_at": _now_iso(),
            "topic": state.get("topic") or state.get("research_topic"),
            "route": state.get("route") or state.get("mode"),
            "lessons": lessons,
        }
        with self.lessons_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        case_dir = self.validation_dir / out_dir.name
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "study_id": out_dir.name,
            "study_dir": str(out_dir),
            "promoted_at": entry["recorded_at"],
            "topic": entry["topic"],
            "route": entry["route"],
            "audit_signature": audit.get("audit_signature"),
        }
        (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return entry

    def _derive_lessons(self, out_dir: Path) -> List[str]:
        """Best-effort, non-LLM notes pulled straight from the run's own artifacts."""
        lessons: List[str] = []

        cases_dir = out_dir / "cases"
        if cases_dir.is_dir():
            for case_dir in sorted(cases_dir.glob("case_*")):
                rr = _read_json(case_dir / "run_result.json")
                if rr and rr.get("rerun_count"):
                    lessons.append(f"{case_dir.name} needed {rr['rerun_count']} rerun(s) before it passed.")
                decision = _read_json(case_dir / "decision.json")
                if decision and decision.get("status") in {"RERUN", "REVISE"}:
                    reason = decision.get("reason", "")
                    lessons.append(f"{case_dir.name}: interpreter flagged {decision['status']} — {reason}".strip())

        mesh_ctx = _read_json(out_dir / "mesh_independence_context.json")
        if mesh_ctx and mesh_ctx.get("escalated_to_gci"):
            lessons.append("Mesh check escalated to Richardson/GCI before a mesh was accepted.")

        return lessons

    # ------------------------------------------------------------- bootstrap

    def list_validation_cases(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in sorted(self.validation_dir.glob("*/manifest.json")):
            data = _read_json(p)
            if data:
                out.append(data)
        return out

    def is_bootstrapped(self, min_studies: int = 3) -> bool:
        """Whether there's enough recorded history to trust a prompt/skill tweak's test.

        Below ``min_studies`` audited studies, :class:`PromptEvolver` is a
        deliberate no-op — there's nothing honest to evaluate a proposed
        change against yet.
        """
        return len(self.list_validation_cases()) >= min_studies

    def recent_lessons(self, n: int = 20) -> List[Dict[str, Any]]:
        if not self.lessons_path.exists():
            return []
        lines = self.lessons_path.read_text(encoding="utf-8").splitlines()
        out: List[Dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    # --------------------------------------------------------- prompt evolution

    def save_variant(self, variant: Dict[str, Any]) -> Path:
        stage = variant.get("stage", "unknown")
        key = variant.get("prompt_key", "unknown")
        digest = hashlib.sha1(
            (variant.get("candidate_prompt") or "").encode("utf-8")
        ).hexdigest()[:10]
        stage_dir = self.variants_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / f"{key}.{digest}.json"
        path.write_text(json.dumps(variant, indent=2, ensure_ascii=False))
        return path

    def promote_variant(self, variant: Dict[str, Any]) -> None:
        """Write a promoted variant into the active-prompts overlay ``PromptLoader`` reads.

        Never overwrites ``prompts/prompts.yaml`` itself — the overlay is
        additive and can always be deleted to fall back to the authoritative
        file, which keeps the full history point (nothing lost, nothing
        silently baked into the source of truth).
        """
        overlay: Dict[str, Any] = {}
        if self.active_prompts_path.exists():
            overlay = yaml.safe_load(self.active_prompts_path.read_text(encoding="utf-8")) or {}
        stage = variant["stage"]
        key = variant["prompt_key"]
        overlay.setdefault(stage, {})[key] = variant["candidate_prompt"]
        self.active_prompts_path.write_text(yaml.safe_dump(overlay, sort_keys=True, allow_unicode=True))

    # -------------------------------------------------------------------- skills

    def list_skills(self) -> List[Dict[str, Any]]:
        manifest = _read_json(self.skills_dir / "manifest.json") or {"skills": []}
        return manifest.get("skills", [])

    def promote_skill(self, skill: Dict[str, Any]) -> None:
        """Add a reusable recipe (mesh spec, solver settings, customModels/ code, ...) to the library.

        ``skill`` should at minimum have ``name``, ``kind``, ``source_study_id``,
        and either inline content or a path under the skill's own directory.
        """
        manifest_path = self.skills_dir / "manifest.json"
        manifest = _read_json(manifest_path) or {"skills": []}
        skill = dict(skill)
        skill.setdefault("promoted_at", _now_iso())
        manifest["skills"].append(skill)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
