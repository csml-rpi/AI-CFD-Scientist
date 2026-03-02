from __future__ import annotations

import json
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class ResultsInterpreterAgent:
    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("ResultsInterpreterAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.1)

    @staticmethod
    def _simple_run_health(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        rc = experiment_result.get("returncode")
        stderr = (experiment_result.get("stderr") or "").lower()
        stdout = (experiment_result.get("stdout") or "").lower()
        has_syntax_signal = any(
            x in (stderr + "\n" + stdout)
            for x in ["fatal", "error", "traceback", "syntax"]
        )

        viz_payload = experiment_result.get("viz") or experiment_result.get(
            "visualization_results"
        )
        viz_ok = True
        if isinstance(viz_payload, dict):
            if viz_payload.get("ok") is False:
                viz_ok = False
            for r in viz_payload.get("results", []):
                if isinstance(r, dict) and not r.get("ok", True):
                    viz_ok = False
                    break

        ok = (rc == 0) and (not has_syntax_signal) and viz_ok
        return {
            "ok": ok,
            "returncode": rc,
            "has_error_signals": has_syntax_signal or (not viz_ok),
            "viz_ok": viz_ok,
        }

    def interpret(
        self,
        idea_json: Dict[str, Any],
        experiment_spec: Dict[str, Any],
        experiment_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_health = self._simple_run_health(experiment_results)

        system = self.prompts.get("system_prompt", "You are CFD Results Interpreter.")
        user_t = self.prompts.get(
            "user_prompt",
            "Study: {idea_json}\nSpec: {experiment_spec}\nResults: {experiment_results}\nReturn JSON.",
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user_t),
        ])
        chain = prompt | self.llm

        content = chain.invoke(
            {
                "idea_json": json.dumps(idea_json),
                "experiment_spec": json.dumps(experiment_spec),
                "experiment_results": json.dumps(experiment_results),
            }
        ).content

        try:
            parsed = json.loads(strip_json_fences(content))
        except Exception:
            parsed = {"raw": content, "parse_error": True}

        # Force a deterministic rerun recommendation when hard errors or bad viz diagnostics are present.
        parsed.setdefault("rerun_required", not base_health["ok"])
        parsed.setdefault("health", base_health)
        if not base_health["ok"]:
            parsed.setdefault(
                "rerun_reason",
                "Execution failed or error signals found in logs or visualization diagnostics.",
            )

        return parsed
