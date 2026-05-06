"""LLM pass to remove unverified references from LaTeX + optional .bib."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.utils import strip_json_fences, strip_latex_fences

_MAIN_TEX_START = "<<<MAIN_TEX_START>>>"
_MAIN_TEX_END = "<<<MAIN_TEX_END>>>"
_BIB_START = "<<<REFERENCES_BIB_START>>>"
_BIB_END = "<<<REFERENCES_BIB_END>>>"


def _extract_delimited_cleanup(content: str) -> Optional[Tuple[str, str]]:
    """Parse delimiter format when the model avoids giant JSON strings."""
    if _MAIN_TEX_START not in content or _MAIN_TEX_END not in content:
        return None
    t = content
    a = t.index(_MAIN_TEX_START) + len(_MAIN_TEX_START)
    b = t.index(_MAIN_TEX_END)
    main_tex = t[a:b].strip("\n")
    bib = ""
    if _BIB_START in t and _BIB_END in t:
        c = t.index(_BIB_START) + len(_BIB_START)
        d = t.index(_BIB_END)
        bib = t[c:d].strip("\n")
    if not main_tex.strip():
        return None
    return main_tex, bib


def _normalize_json_object_slice(blob: str) -> str:
    """Turn common LLM mistakes (single-quoted keys) into strict JSON keys."""
    s = blob.strip()
    # Only touch top-level keys we expect.
    s = re.sub(r"'(main_tex|references_bib)'\s*:", r'"\1":', s, count=2)
    return s


def _parse_cleanup_payload(content: str) -> Dict[str, Any]:
    """
    Accept STRICT JSON or delimiter-wrapped body. Raises ValueError if nothing parses.
    """
    raw = content if isinstance(content, str) else str(content)
    cleaned = strip_json_fences(raw.strip())
    cleaned = strip_latex_fences(cleaned)

    delim = _extract_delimited_cleanup(cleaned)
    if delim is not None:
        return {"main_tex": delim[0], "references_bib": delim[1]}

    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s == -1 or e <= s:
        raise ValueError("reference cleanup LLM did not return JSON object or delimiters")

    slice_ = cleaned[s : e + 1]
    try:
        payload = json.loads(slice_)
    except json.JSONDecodeError:
        payload = json.loads(_normalize_json_object_slice(slice_))
    if not isinstance(payload, dict):
        raise ValueError("reference cleanup payload is not an object")
    return payload


def cleanup_hallucinated_references(
    *,
    main_tex: str,
    references_bib: str,
    hallucinated: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.1,
) -> Tuple[str, str]:
    """
    Returns (revised_main_tex, revised_references_bib).

    hallucinated items should include at least {"key": "...", "reason": "..."}.
    """
    if not hallucinated:
        return main_tex, references_bib

    system = (
        "You are a meticulous LaTeX editor. You remove invalid or hallucinated citations "
        "while keeping all other content, notation, figures, and valid citations intact."
    )
    user = (
        "The following bibliography keys were flagged as unverified or hallucinated (do not cite them):\n"
        "{bad_json}\n\n"
        "TASK:\n"
        "1) Remove every \\bibitem{{key}} block for those keys from \\begin{{thebibliography}}...\\end{{thebibliography}} if present.\n"
        "2) Remove those keys from every \\cite, \\citep, \\citet, \\citeauthor, etc. "
        "If a \\cite{{a,b}} mixes good and bad keys, drop only the bad keys and keep the good ones. "
        "If removing keys leaves an empty \\cite{{}}, delete the entire \\cite command including following punctuation "
        "adjustment so the sentence remains grammatical (no double spaces; fix '()' leftovers).\n"
        "3) If the narrative now claims results 'supported by' removed work, soften or delete that clause without adding new citations.\n"
        "4) If references.bib content is non-empty: delete the @ entries matching those keys only.\n"
        "5) Do not add new references, packages, or structural changes beyond what is needed.\n"
        "6) If the LaTeX is accidentally wrapped in markdown fences (e.g. leading ```latex), remove those fences.\n\n"
        "OUTPUT FORMAT (choose exactly one):\n"
        "(A) STRICT JSON only: an object with keys \"main_tex\" and \"references_bib\" (string values). "
        "Every key and string must use double quotes per JSON; escape internal quotes and backslashes. No markdown fences.\n"
        "(B) Plain delimiter format (use if JSON would be fragile). Emit exactly these lines, nothing before the first marker:\n"
        f"{_MAIN_TEX_START}\n"
        "(full LaTeX document)\n"
        f"{_MAIN_TEX_END}\n"
        f"{_BIB_START}\n"
        "(full .bib body, or empty)\n"
        f"{_BIB_END}\n\n"
        "CURRENT main.tex (or full document):\n"
        "{main_tex}\n\n"
        "CURRENT references.bib (may be empty):\n"
        "{references_bib}\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", user),
        ]
    )
    llm = create_langchain_llm(model=model, temperature=temperature)
    chain = prompt | llm
    invoke_kw = {
        "bad_json": json.dumps(hallucinated, indent=2)[:12000],
        "main_tex": main_tex[:120000],
        "references_bib": references_bib[:80000],
    }

    def _run_once() -> str:
        out = chain.invoke(invoke_kw)
        return getattr(out, "content", str(out))

    content = _run_once()
    try:
        payload = _parse_cleanup_payload(content)
    except (ValueError, json.JSONDecodeError) as first_err:
        # One retry: models often emit Python-ish dicts or truncated JSON on long TeX.
        strict_tail = (
            "\n\nCRITICAL: Your previous reply could not be parsed. "
            "Use format (B) delimiters only. Copy the full edited LaTeX between "
            f"{_MAIN_TEX_START} and {_MAIN_TEX_END}, then bib between "
            f"{_BIB_START} and {_BIB_END}. No JSON, no markdown fences."
        )
        prompt2 = ChatPromptTemplate.from_messages(
            [
                ("system", system + strict_tail),
                ("human", user + strict_tail),
            ]
        )
        chain2 = prompt2 | llm
        out2 = chain2.invoke(invoke_kw)
        content2 = getattr(out2, "content", str(out2))
        try:
            payload = _parse_cleanup_payload(content2)
        except (ValueError, json.JSONDecodeError) as e2:
            raise ValueError(
                "reference cleanup LLM output was not valid JSON or delimiter format "
                f"(first error: {first_err}; retry error: {e2})"
            ) from e2

    new_tex = str(payload.get("main_tex", ""))
    new_bib = str(payload.get("references_bib", ""))
    if not new_tex.strip():
        raise ValueError("reference cleanup returned empty main_tex")
    return new_tex, new_bib


def load_model_from_settings() -> str:
    from cfd_langgraph.config import get_settings

    return get_settings().model


def sync_body_tex(paper_dir: Path, main_tex: str) -> None:
    body = paper_dir / "sections" / "body.tex"
    if body.is_file():
        body.write_text(main_tex, encoding="utf-8")
