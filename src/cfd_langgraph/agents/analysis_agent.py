from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from cfd_langchain.llm.factory import create_langchain_llm
from cfd_langchain.utils import strip_json_fences


class AnalysisAgent:
    def __init__(self, model: str):
        self.model = model
        self.llm = create_langchain_llm(model=model, temperature=0.0)
        self.plot_planner_llm = create_langchain_llm(model=model, temperature=0.1)

    def analyze_text_bundle(self, batch_name: str, bundle_text: str, extra_context: Optional[str] = None) -> str:
        system = "You are a CFD expert specializing in simulation diagnostics and scientific analysis."
        user = (
            "Analyze this CFD batch named {batch_name}.\n"
            "Context:\n{extra_context}\n\n"
            "Bundle:\n{bundle_text}\n\n"
            "Return detailed per-case + cross-case analysis, observations, conclusions, and publication-ready figure suggestions."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user),
        ])
        chain = prompt | self.llm
        return chain.invoke({"batch_name": batch_name, "bundle_text": bundle_text, "extra_context": extra_context or ""}).content

    def save_analysis(self, out_path: Path, text: str) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    # -------------------------
    # Generic plot orchestration
    # -------------------------
    def _plan_plot_jobs(self, user_request: str, available_arrays: List[str]) -> Dict[str, Any]:
        system = (
            "You are a CFD visualization planner. Given a plotting request and available scalar/vector arrays, "
            "create a compact plot plan. Use only arrays that exist."
        )
        user = (
            "Request:\n{request}\n\n"
            "Available arrays:\n{arrays}\n\n"
            "Return ONLY JSON with schema:\n"
            "{\n"
            "  \"plots\": [\n"
            "    {\n"
            "      \"type\": \"contour|slice|streamlines|glyph|volume|line|threshold|isosurface\",\n"
            "      \"name\": \"string\",\n"
            "      \"scalar\": \"optional string\",\n"
            "      \"params\": {\"any\": \"json\"}\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        chain = prompt | self.plot_planner_llm
        raw = chain.invoke({"request": user_request, "arrays": ", ".join(available_arrays)}).content
        try:
            parsed = json.loads(strip_json_fences(raw))
            if not isinstance(parsed, dict):
                raise ValueError("not dict")
            parsed.setdefault("plots", [])
            return parsed
        except Exception:
            # Safe fallback
            fallback = {
                "plots": [
                    {
                        "type": "slice",
                        "name": "slice_default",
                        "scalar": available_arrays[0] if available_arrays else None,
                        "params": {"origin": None, "normal": [0, 0, 1]},
                    }
                ]
            }
            return fallback

    @staticmethod
    def _screenshot(plotter, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out_path))
        plotter.close()

    def _render_plot(self, mesh, plot_job: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
        import pyvista as pv  # type: ignore

        ptype = (plot_job.get("type") or "slice").lower()
        scalar = plot_job.get("scalar")
        params = plot_job.get("params") or {}

        try:
            plotter = pv.Plotter(off_screen=True)

            if ptype == "contour" or ptype == "isosurface":
                n = int(params.get("n", 15))
                if scalar and scalar in mesh.array_names:
                    obj = mesh.contour(isosurfaces=n, scalars=scalar)
                else:
                    obj = mesh.outline()
                plotter.add_mesh(obj, cmap=params.get("cmap", "viridis"))

            elif ptype == "slice":
                normal = params.get("normal", [0, 0, 1])
                origin = params.get("origin", None)
                obj = mesh.slice(normal=normal, origin=origin)
                plotter.add_mesh(obj, scalars=scalar if scalar in mesh.array_names else None, cmap=params.get("cmap", "plasma"))

            elif ptype == "threshold":
                value = params.get("value", None)
                obj = mesh.threshold(value=value, scalars=scalar) if (scalar and scalar in mesh.array_names) else mesh
                plotter.add_mesh(obj, scalars=scalar if scalar in mesh.array_names else None)

            elif ptype == "streamlines":
                vectors = params.get("vectors")
                if vectors and vectors in mesh.array_names:
                    seeds = mesh.slice(normal=params.get("seed_normal", [0, 0, 1]))
                    obj = mesh.streamlines_from_source(seeds, vectors=vectors)
                    plotter.add_mesh(obj, color="white")
                else:
                    plotter.add_mesh(mesh.outline(), color="white")

            elif ptype == "glyph":
                vectors = params.get("vectors")
                orient = vectors if vectors in mesh.array_names else None
                obj = mesh.glyph(orient=orient, factor=float(params.get("factor", 0.1)))
                plotter.add_mesh(obj)

            elif ptype == "volume":
                if scalar and scalar in mesh.array_names:
                    plotter.add_volume(mesh, scalars=scalar, cmap=params.get("cmap", "viridis"))
                else:
                    plotter.add_mesh(mesh.outline())

            elif ptype == "line":
                # For line plots we sample and use matplotlib output
                import matplotlib.pyplot as plt  # type: ignore

                p0 = params.get("p0", [0, 0, 0])
                p1 = params.get("p1", [1, 0, 0])
                n = int(params.get("n", 200))
                sampled = mesh.sample_over_line(p0, p1, resolution=n)
                if scalar and scalar in sampled.array_names:
                    y = sampled[scalar]
                else:
                    # fallback to first array
                    arrs = sampled.array_names
                    y = sampled[arrs[0]] if arrs else [0] * (n + 1)
                    scalar = arrs[0] if arrs else "value"
                x = list(range(len(y)))

                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(7, 4), dpi=120)
                plt.plot(x, y, linewidth=2)
                plt.title(plot_job.get("name", "line_plot"))
                plt.xlabel("sample index")
                plt.ylabel(str(scalar))
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
                return {"ok": True, "output": str(out_path), "type": ptype}

            else:
                plotter.add_mesh(mesh.outline())

            plotter.add_axes()
            self._screenshot(plotter, out_path)
            return {"ok": True, "output": str(out_path), "type": ptype}

        except Exception as e:
            return {"ok": False, "error": str(e), "type": ptype}

    def generate_plots_from_foam_data(
        self,
        foam_data_path: Path,
        request_text: str,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """
        Modular post-run plotting:
        - loads foam/VTK data using PyVista
        - asks LLM to decide suitable plot set
        - renders requested plot types and saves figures
        """
        try:
            import pyvista as pv  # type: ignore
        except Exception as e:
            return {"ok": False, "error": f"PyVista import failed: {e}"}

        try:
            mesh = pv.read(str(foam_data_path))
        except Exception as e:
            return {"ok": False, "error": f"Failed to read data via PyVista: {e}"}

        arrays = list(getattr(mesh, "array_names", []))
        plan = self._plan_plot_jobs(request_text, arrays)

        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, job in enumerate(plan.get("plots", []), 1):
            name = job.get("name", f"plot_{i:02d}")
            out_path = out_dir / f"{i:02d}_{name}.png"
            results.append(self._render_plot(mesh, job, out_path))

        return {
            "ok": True,
            "foam_data": str(foam_data_path),
            "available_arrays": arrays,
            "plan": plan,
            "results": results,
        }
