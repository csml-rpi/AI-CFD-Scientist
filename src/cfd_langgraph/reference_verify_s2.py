"""
Semantic Scholar–backed reference verification (s2-title-to-bibtex style).

Resolves each bibliography entry via DOI (preferred) or title search and scores
consistency with the manuscript entry. Optionally trusts titles present in the
study's lit.json corpus.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# --- normalization / parsing -------------------------------------------------

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.I)


def normalize_title(title: str) -> str:
    t = title.replace("\n", " ").strip().lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return float(SequenceMatcher(None, na, nb).ratio())


def normalize_doi(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    s = s.rstrip(" .)")
    return s


_CITE_RE = re.compile(
    r"\\cite[tpex]*?(?:\[[^\]]*\])?\{([^}]+)\}",
    re.MULTILINE,
)
_BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
_BIBTYPE_RE = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")


def extract_cite_keys(tex: str) -> Set[str]:
    keys: Set[str] = set()
    for m in _CITE_RE.finditer(tex):
        inner = m.group(1)
        for part in inner.split(","):
            k = part.strip()
            if k:
                keys.add(k)
    return keys


def _extract_doi(block: str) -> Optional[str]:
    m = re.search(r"DOI:\s*([^\s\\}]+)", block, re.I)
    if m:
        return normalize_doi(m.group(1))
    m = re.search(r"doi\.org/([^\s}\]]+)", block, re.I)
    if m:
        return normalize_doi(m.group(1))
    return None


def _extract_latex_title(block: str) -> str:
    """Heuristic: first `` ... '' span, else first quoted segment."""
    m = re.search(r"``((?:.|\n)*?)''", block)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(r"``\s*((?:.|\n)*?)\s*''", block)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def parse_thebibliography(tex: str) -> List[Dict[str, Any]]:
    start = tex.find(r"\begin{thebibliography}")
    end = tex.find(r"\end{thebibliography}")
    if start == -1 or end == -1 or end <= start:
        return []
    block = tex[start:end]
    items: List[Dict[str, Any]] = []
    for m in _BIBITEM_RE.finditer(block):
        key = m.group(1).strip()
        start_pos = m.end()
        next_m = _BIBITEM_RE.search(block, start_pos)
        end_pos = next_m.start() if next_m else len(block)
        body = block[start_pos:end_pos].strip()
        items.append(
            {
                "key": key,
                "raw": body,
                "doi": _extract_doi(body),
                "title_guess": _extract_latex_title(body),
            }
        )
    return items


def parse_bibtex_keys_and_blocks(bib_text: str) -> List[Dict[str, Any]]:
    if not bib_text.strip():
        return []
    entries: List[Dict[str, Any]] = []
    for m in re.finditer(r"(@\w+\{[^@]*)", bib_text, re.DOTALL):
        chunk = m.group(1)
        km = re.match(r"@\w+\{\s*([^,\s]+)\s*,", chunk)
        if not km:
            continue
        key = km.group(1).strip()
        title_m = re.search(r"title\s*=\s*\{([^}]*)\}", chunk, re.I | re.DOTALL)
        title = title_m.group(1).replace("\n", " ").strip() if title_m else ""
        doi_m = re.search(r"doi\s*=\s*\{([^}]+)\}", chunk, re.I)
        doi = normalize_doi(doi_m.group(1)) if doi_m else None
        entries.append({"key": key, "raw": chunk.strip(), "doi": doi, "title_guess": title})
    return entries


def load_literature_titles(lit_path: Optional[Path]) -> Tuple[List[str], List[str]]:
    """Returns (raw_titles, normalized_titles)."""
    if not lit_path or not lit_path.is_file():
        return [], []
    try:
        data = json.loads(lit_path.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    records: List[Any]
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("papers"), list):
        records = data["papers"]
    else:
        records = []
    raw: List[str] = []
    for r in records:
        if isinstance(r, dict):
            t = str(r.get("title", "")).strip()
            if t:
                raw.append(t)
    norm = [normalize_title(t) for t in raw]
    return raw, norm


def literature_match_score(title_guess: str, lit_norm: List[str]) -> float:
    if not title_guess or not lit_norm:
        return 0.0
    ng = normalize_title(title_guess)
    if not ng:
        return 0.0
    return max((SequenceMatcher(None, ng, ln).ratio() for ln in lit_norm if ln), default=0.0)


# --- Semantic Scholar ---------------------------------------------------------

S2_GRAPH = "https://api.semanticscholar.org/graph/v1"


def _s2_headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    key = os.environ.get("S2_API_KEY", "").strip()
    if key:
        h["x-api-key"] = key
    return h


def s2_get_paper_by_doi(doi: str, timeout: float = 60.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    enc = urllib.parse.quote(doi, safe="")
    url = f"{S2_GRAPH}/paper/DOI:{enc}"
    params = {"fields": "title,authors,year,externalIds,url"}
    try:
        r = requests.get(url, params=params, headers=_s2_headers(), timeout=timeout)
        if r.status_code == 404:
            return None, "doi_not_found_on_semantic_scholar"
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, f"s2_request_error:{e}"


def s2_search_title(title: str, limit: int = 5, timeout: float = 60.0) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not title or len(title) < 8:
        return [], "title_too_short"
    url = f"{S2_GRAPH}/paper/search"
    # Short query like lit.py
    words = [w for w in normalize_title(title).split() if len(w) > 2][:12]
    query = " ".join(words)
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,externalIds,url",
    }
    try:
        r = requests.get(url, params=params, headers=_s2_headers(), timeout=timeout)
        r.raise_for_status()
        data = r.json() or {}
        return list(data.get("data") or []), None
    except requests.RequestException as e:
        return [], f"s2_search_error:{e}"


@dataclass
class VerifyEntryResult:
    key: str
    source: str  # "thebibliography" | "bibfile"
    verified: bool
    reason: str
    title_guess: str = ""
    s2_title: str = ""
    similarity: float = 0.0
    from_study_literature: bool = False


def verify_bibliography_entry(
    entry: Dict[str, Any],
    *,
    source: str,
    lit_norm: List[str],
    literature_title_threshold: float = 0.88,
    doi_title_threshold: float = 0.72,
    search_title_threshold: float = 0.78,
    s2_sleep_s: float = 0.25,
) -> VerifyEntryResult:
    key = str(entry.get("key", ""))
    title_guess = str(entry.get("title_guess", "")).strip()
    doi = entry.get("doi")

    lit_score = literature_match_score(title_guess, lit_norm)
    if lit_score >= literature_title_threshold:
        time.sleep(s2_sleep_s)
        return VerifyEntryResult(
            key=key,
            source=source,
            verified=True,
            reason="matches_study_lit_json_title",
            title_guess=title_guess,
            similarity=lit_score,
            from_study_literature=True,
        )

    if doi:
        paper, err = s2_get_paper_by_doi(doi)
        time.sleep(s2_sleep_s)
        if err:
            return VerifyEntryResult(
                key=key,
                source=source,
                verified=False,
                reason=err or "doi_lookup_failed",
                title_guess=title_guess,
            )
        s2_title = str(paper.get("title", "") if paper else "")
        sim = title_similarity(title_guess, s2_title) if title_guess else title_similarity(entry.get("raw", ""), s2_title)
        if sim >= doi_title_threshold:
            return VerifyEntryResult(
                key=key,
                source=source,
                verified=True,
                reason="doi_resolves_semantic_scholar_title_match",
                title_guess=title_guess,
                s2_title=s2_title,
                similarity=sim,
            )
        return VerifyEntryResult(
            key=key,
            source=source,
            verified=False,
            reason=f"doi_resolves_but_title_mismatch(sim={sim:.2f})",
            title_guess=title_guess,
            s2_title=s2_title,
            similarity=sim,
        )

    if not title_guess:
        return VerifyEntryResult(
            key=key,
            source=source,
            verified=False,
            reason="no_doi_and_no_parsable_title",
            title_guess=title_guess,
        )

    hits, err = s2_search_title(title_guess)
    time.sleep(s2_sleep_s)
    if err or not hits:
        return VerifyEntryResult(
            key=key,
            source=source,
            verified=False,
            reason=err or "s2_search_no_results",
            title_guess=title_guess,
        )
    best = hits[0]
    s2_title = str(best.get("title", ""))
    sim = title_similarity(title_guess, s2_title)
    if sim >= search_title_threshold:
        return VerifyEntryResult(
            key=key,
            source=source,
            verified=True,
            reason="s2_title_search_top_hit_match",
            title_guess=title_guess,
            s2_title=s2_title,
            similarity=sim,
        )
    return VerifyEntryResult(
        key=key,
        source=source,
        verified=False,
        reason=f"s2_top_hit_low_similarity(sim={sim:.2f})",
        title_guess=title_guess,
        s2_title=s2_title,
        similarity=sim,
    )


def collect_tex_bundle(paper_dir: Path) -> str:
    parts: List[str] = []
    main = paper_dir / "main.tex"
    if main.is_file():
        parts.append(main.read_text(encoding="utf-8", errors="ignore"))
    sec = paper_dir / "sections"
    if sec.is_dir():
        for p in sorted(sec.glob("*.tex")):
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def build_verification_report(
    paper_dir: Path,
    lit_path: Optional[Path],
    *,
    literature_title_threshold: float = 0.88,
    doi_title_threshold: float = 0.72,
    search_title_threshold: float = 0.78,
) -> Dict[str, Any]:
    tex = collect_tex_bundle(paper_dir)
    cite_keys = extract_cite_keys(tex)
    _lit_raw, lit_norm = load_literature_titles(lit_path)

    bib_path = paper_dir / "references.bib"
    bib_text = bib_path.read_text(encoding="utf-8", errors="ignore") if bib_path.is_file() else ""

    bib_entries = parse_bibtex_keys_and_blocks(bib_text)
    bib_items = parse_thebibliography(tex)

    results: List[VerifyEntryResult] = []
    seen_keys: Set[str] = set()

    for ent in bib_items:
        seen_keys.add(ent["key"])
        results.append(
            verify_bibliography_entry(
                ent,
                source="thebibliography",
                lit_norm=lit_norm,
                literature_title_threshold=literature_title_threshold,
                doi_title_threshold=doi_title_threshold,
                search_title_threshold=search_title_threshold,
            )
        )

    for ent in bib_entries:
        if ent["key"] in seen_keys:
            continue
        seen_keys.add(ent["key"])
        results.append(
            verify_bibliography_entry(
                ent,
                source="bibfile",
                lit_norm=lit_norm,
                literature_title_threshold=literature_title_threshold,
                doi_title_threshold=doi_title_threshold,
                search_title_threshold=search_title_threshold,
            )
        )

    bad = [r for r in results if not r.verified]
    bad_keys = [r.key for r in bad]

    missing_bib = sorted(cite_keys - seen_keys)
    orphan_bib = sorted(seen_keys - cite_keys)

    return {
        "paper_dir": str(paper_dir.resolve()),
        "literature_path": str(lit_path) if lit_path else "",
        "literature_title_sample": _lit_raw[:8],
        "counts": {
            "cite_keys": len(cite_keys),
            "bibliography_entries": len(seen_keys),
            "verified": sum(1 for r in results if r.verified),
            "unverified": len(bad),
            "missing_bibliography_for_cite": len(missing_bib),
        },
        "missing_bib_keys_for_cites": missing_bib,
        "orphan_bib_keys": orphan_bib,
        "entries": [
            {
                "key": r.key,
                "source": r.source,
                "verified": r.verified,
                "reason": r.reason,
                "title_guess": r.title_guess,
                "s2_title": r.s2_title,
                "similarity": r.similarity,
                "from_study_literature": r.from_study_literature,
            }
            for r in results
        ],
        "hallucination_candidates": [
            {
                "key": r.key,
                "reason": r.reason,
                "title_guess": r.title_guess,
                "s2_title": r.s2_title,
            }
            for r in bad
        ],
    }


def merge_missing_cites_as_hallucinations(report: Dict[str, Any]) -> Dict[str, Any]:
    """Cites without any bib entry are treated as cleanup targets."""
    extras = []
    for k in report.get("missing_bib_keys_for_cites", []) or []:
        extras.append(
            {
                "key": k,
                "reason": "cited_but_no_bibliography_entry",
                "title_guess": "",
                "s2_title": "",
            }
        )
    hc = list(report.get("hallucination_candidates", []) or [])
    existing = {h["key"] for h in hc if isinstance(h, dict)}
    for e in extras:
        if e["key"] not in existing:
            hc.append(e)
    report = dict(report)
    report["hallucination_candidates"] = hc
    report["cleanup_keys"] = sorted({h["key"] for h in hc if isinstance(h, dict)})
    return report
