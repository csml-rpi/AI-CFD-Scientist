#!/usr/bin/env python3
"""
Drift checker — verifies every prompt claimed embedded in cfd-skills/*/SKILL.md
matches the verbatim text in prompts/prompts.yaml byte-for-byte.

Strategy:
  1. Load prompts/prompts.yaml.
  2. Walk every SKILL.md under cfd-skills/.
  3. For each header line of the form
       (from `prompts/prompts.yaml: AGENT.KEY`)
     check that yaml[AGENT][KEY].rstrip() appears verbatim somewhere in the
     SKILL.md text. (We use substring containment to dodge markdown-fence
     boundary issues caused by prompts that themselves contain ``` fences.)
  4. Also verify openfoam_literature_change_agent_prompt_v2.txt is fully
     mirrored verbatim inside cfd-skills/cfd-code-modify/SKILL.md.

Exit 0 → all embeds match. Exit 1 → at least one mismatch.

This is a maintenance tool, not part of the runtime pipeline. Run it after
editing prompts/prompts.yaml or any SKILL.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_YAML = REPO_ROOT / "prompts" / "prompts.yaml"
CODE_MOD_PROTOCOL = REPO_ROOT / "openfoam_literature_change_agent_prompt_v2.txt"
SKILLS_DIR = REPO_ROOT / "cfd-skills"

REF_RE = re.compile(r"from `prompts/prompts\.yaml: ([A-Za-z][A-Za-z0-9_]*)\.([a-z][a-z0-9_]*)`")


def normalize(text: str) -> str:
    """Light normalization: trim trailing whitespace per line + collapse trailing newlines."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def load_yaml() -> dict[str, dict[str, str]]:
    with PROMPTS_YAML.open() as f:
        return yaml.safe_load(f)


def find_skill_refs(skill_path: Path) -> list[tuple[str, str]]:
    """Return list of (Agent, key) declared embedded in this SKILL.md."""
    text = skill_path.read_text()
    return list({(m.group(1), m.group(2)) for m in REF_RE.finditer(text)})


def check_skill(skill_path: Path, prompts: dict[str, dict[str, str]]) -> list[str]:
    """Return list of mismatch descriptions. Empty list = OK."""
    refs = find_skill_refs(skill_path)
    if not refs:
        return []
    skill_text = skill_path.read_text()
    skill_norm = normalize(skill_text)
    failures: list[str] = []
    for agent, key in sorted(refs):
        if agent not in prompts:
            failures.append(f"  MISSING_AGENT: {agent}.{key} (agent not in YAML)")
            continue
        if key not in prompts[agent]:
            failures.append(f"  MISSING_KEY: {agent}.{key}")
            continue
        yaml_text = prompts[agent][key]
        if not isinstance(yaml_text, str):
            failures.append(f"  NOT_STRING: {agent}.{key} (type={type(yaml_text).__name__})")
            continue
        yaml_norm = normalize(yaml_text)
        if yaml_norm in skill_norm:
            continue
        # Try a looser whitespace-normalized comparison to localize where it diverges.
        diff_idx = first_diff_index(yaml_norm, skill_norm)
        failures.append(
            f"  MISMATCH: {agent}.{key}\n"
            f"    yaml_chars={len(yaml_norm)}, first_divergence_at_yaml_offset={diff_idx}\n"
            f"    yaml_excerpt: {yaml_norm[max(0, diff_idx-40):diff_idx+80]!r}"
        )
    return failures


def first_diff_index(needle: str, haystack: str) -> int:
    """If needle isn't a substring of haystack, find the longest prefix that IS,
    and return that prefix's length (the offset inside `needle` where divergence starts).
    """
    if not needle:
        return 0
    lo, hi = 0, len(needle)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if needle[:mid] in haystack:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def check_code_mod_protocol() -> list[str]:
    """Verify openfoam_literature_change_agent_prompt_v2.txt is mirrored
    verbatim inside cfd-code-modify/SKILL.md."""
    failures: list[str] = []
    skill_path = SKILLS_DIR / "cfd-code-modify" / "SKILL.md"
    if not skill_path.exists():
        return ["  MISSING: cfd-code-modify/SKILL.md"]
    if not CODE_MOD_PROTOCOL.exists():
        return ["  MISSING: openfoam_literature_change_agent_prompt_v2.txt"]
    proto = normalize(CODE_MOD_PROTOCOL.read_text())
    skill = normalize(skill_path.read_text())
    if proto in skill:
        return []
    diff_idx = first_diff_index(proto, skill)
    failures.append(
        f"  MISMATCH: openfoam_literature_change_agent_prompt_v2.txt vs cfd-code-modify/SKILL.md\n"
        f"    proto_chars={len(proto)}, first_divergence_at_proto_offset={diff_idx}\n"
        f"    proto_excerpt: {proto[max(0, diff_idx-40):diff_idx+80]!r}"
    )
    return failures


def main() -> int:
    prompts = load_yaml()
    print(f"YAML loaded: {sum(len(v) for v in prompts.values() if isinstance(v, dict))} prompts across {len(prompts)} agents.")
    print(f"Scanning {SKILLS_DIR}...")
    total_refs = 0
    total_failures: list[tuple[str, list[str]]] = []
    for skill_path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        refs = find_skill_refs(skill_path)
        if refs:
            total_refs += len(refs)
            print(f"\n{skill_path.relative_to(REPO_ROOT)}: {len(refs)} embedded prompts")
        failures = check_skill(skill_path, prompts)
        if failures:
            total_failures.append((str(skill_path.relative_to(REPO_ROOT)), failures))

    print("\n=== openfoam_literature_change_agent_prompt_v2.txt mirror check ===")
    proto_failures = check_code_mod_protocol()
    if proto_failures:
        total_failures.append(("openfoam_literature_change_agent_prompt_v2.txt", proto_failures))
        print(" FAIL")
    else:
        print(" OK")

    print()
    print(f"Total embedded prompt references checked: {total_refs}")
    if total_failures:
        print(f"FAIL: {sum(len(f) for _, f in total_failures)} mismatch(es) across {len(total_failures)} file(s):")
        for path, failures in total_failures:
            print(f"\n{path}:")
            for f in failures:
                print(f)
        return 1
    print("PASS: every embedded prompt matches prompts/prompts.yaml byte-for-byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
