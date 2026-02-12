import os
import json
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.base_llm import create_client, get_response_from_llm


class SelectorAgent:
    """Select a subset of candidate experiments to best test a hypothesis.

    This agent is intentionally narrow:
    - Input: hypothesis text + ideation JSON + explicit candidate experiment list
    - Output: up to K selected candidate IDs (plus brief reasons)

    Selection criterion (only): how strongly each experiment can support OR reject the hypothesis.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.client, self.model_id = create_client(model)
        self.prompts = self._load_prompts()
        print(f"SelectorAgent initialized with model: {self.model_id}")

    def _load_prompts(self) -> Dict[str, str]:
        try:
            prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.yaml")
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = yaml.safe_load(f)
            return prompts_data.get("SelectorAgent", {}) or {}
        except Exception as e:
            print(f"Warning: Could not load SelectorAgent prompts: {e}")
            return {}

    @staticmethod
    def _extract_json_object(text: str) -> str:
        if not text:
            raise ValueError("Empty model response; expected JSON")
        s = text.strip()
        # Strip fenced code blocks if present
        if s.startswith("```"):
            s = s.split("```", 2)[1] if "```" in s else s
            s = s.replace("json", "", 1).strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not find JSON object in response: {s[:200]}")
        return s[start : end + 1]

    def select_experiments(
        self,
        *,
        hypothesis_text: str,
        idea_json: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        k: int = 10,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        system_prompt = self.prompts.get("system_prompt", "").strip()
        user_prompt_tmpl = self.prompts.get("user_prompt", "").strip()
        if not system_prompt or not user_prompt_tmpl:
            raise ValueError("SelectorAgent prompts missing system_prompt or user_prompt in prompts.yaml")

        # Keep payload compact and structured. Avoid dumping full ideation prompt text.
        idea_summary = {
            "study_id": idea_json.get("study_id"),
            "description": idea_json.get("description"),
            "solver": idea_json.get("solver"),
            "target_CFL": idea_json.get("target_CFL"),
            "post": idea_json.get("post"),
            "num_cases": len(idea_json.get("cases", []) or []) if isinstance(idea_json, dict) else None,
        }

        payload = {
            "hypothesis": hypothesis_text,
            "k": int(k),
            "idea": idea_summary,
            "candidates": candidates,
        }

        user_prompt = user_prompt_tmpl.replace("{payload_json}", json.dumps(payload, indent=2))

        content, _ = get_response_from_llm(
            prompt=user_prompt,
            client=self.client,
            model=self.model_id,
            system_message=system_prompt,
            temperature=temperature,
            print_debug=False,
        )

        json_text = self._extract_json_object(content)
        out = json.loads(json_text)
        if not isinstance(out, dict):
            raise ValueError("SelectorAgent output must be a JSON object")
        return out
