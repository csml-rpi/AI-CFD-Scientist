from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel


_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model: str = os.environ.get(
        "CFD_SCIENTIST_MODEL",
        os.environ.get("CFD_SCIENITST_MODEL", "us.anthropic.claude-sonnet-4-6"),
    )
    # Explicit provider override. If unset, provider is inferred from `model` string.
    # Foam-Agent also supports its own provider env vars; our runner maps CFD_* values into FOAMAGENT_*.
    llm_provider: str = (
        os.environ.get("CFD_SCIENTIST_LLM_PROVIDER")
        or os.environ.get("CFD_SCIEINTIST_LLM_PROVIDER")
        or "auto"
    ).lower()
    prompts_path: Path = Path(
        os.environ.get(
            "CFD_PROMPTS_PATH",
            str(_REPO_ROOT / "prompts" / "prompts.yaml"),
        )
    )
    foam_agent_main: Path = Path(
        os.environ.get(
            "FOAM_AGENT_MAIN",
            str(_REPO_ROOT / "Foam-Agent" / "foambench_main.py"),
        )
    )
    openfoam_path: str = os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10")

    # Literature-aware ideation
    s2_api_key: str | None = os.environ.get("S2_API_KEY")
    brave_api_key: str | None = os.environ.get("BRAVE_SEARCH_API_KEY")
    ideation_enable_literature: bool = os.environ.get("CFD_IDEATION_ENABLE_LITERATURE", "1") == "1"
    ideation_max_papers: int = int(os.environ.get("CFD_IDEATION_MAX_PAPERS", "40"))
    ideation_max_web_results: int = int(os.environ.get("CFD_IDEATION_MAX_WEB_RESULTS", "5"))
    ideation_max_experiments: int = int(os.environ.get("CFD_IDEATION_MAX_EXPERIMENTS", "10"))

    # Strict novelty gate for ideation output vs retrieved prior studies
    ideation_novelty_threshold: float = float(os.environ.get("CFD_IDEATION_NOVELTY_THRESHOLD", "0.62"))
    ideation_novelty_max_retries: int = int(os.environ.get("CFD_IDEATION_NOVELTY_MAX_RETRIES", "3"))

    # Lighter model for hypothesis validation/repair loop (defaults to main model if unset).
    # Set CFD_SCIENTIST_VALIDATOR_MODEL to a fast model (e.g. Haiku) to speed up the
    # validate+repair loop without affecting requirement generation quality.
    validator_model: str = os.environ.get("CFD_SCIENTIST_VALIDATOR_MODEL", "")

    # Orchestration controls
    workflow_max_experiments_total: int = int(os.environ.get("CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL", "50"))
    workflow_max_reruns_per_experiment: int = int(os.environ.get("CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT", "10"))



def get_settings() -> Settings:
    return Settings()
