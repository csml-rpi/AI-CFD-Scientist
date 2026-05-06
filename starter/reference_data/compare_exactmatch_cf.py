#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare benchmark-style periodic-hill SA case against local exact-match Cf reference.")
    ap.add_argument("--case", type=Path, default=Path("."), help="OpenFOAM case directory")
    ap.add_argument("--time", type=str, default=None, help="Time directory. Default: latest with wallShearStress")
    ap.add_argument("--reference", type=Path, default=Path("reference_exactmatch_cf.csv"), help="Local exact-match reference CSV")
    ap.add_argument("--out", type=Path, default=None, help="Output directory. Default: <case>/comparison_exactmatch/<time>")
    return ap.parse_args()


def read_text(path: Path) -> str:
    return path.read_text()


def latest_time_with_field(case_dir: Path, field_name: str) -> Path:
    times = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            value = float(entry.name)
        except ValueError:
            continue
        if (entry / field_name).exists():
            times.append((value, entry))
    if not times:
        raise FileNotFoundError(f"No time directories with {field_name} found in {case_dir}")
    return sorted(times)[-1][1]


def parse_scalar_property(text: str, name: str) -> float:
    match = re.search(rf"\b{name}\b\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;", text)
    if not match:
        raise ValueError(f"Could not find scalar property {name}")
    return float(match.group(1))


def parse_points(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "(")
    pts = []
    for line in lines[start + 1:]:
        s = line.strip()
        if s == ")":
            break
        pts.append([float(v) for v in s.strip("()").split()])
    return np.array(pts)


def parse_faces(path: Path) -> list[list[int]]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "(")
    faces = []
    for line in lines[start + 1:]:
        s = line.strip()
        if s == ")":
            break
        m = re.match(r"(\d+)\((.*)\)", s)
        faces.append([int(v) for v in m.group(2).split()])
    return faces


def parse_patch_info(boundary_path: Path, patch_name: str) -> tuple[int, int]:
    text = boundary_path.read_text()
    match = re.search(
        rf"\b{patch_name}\b\s*\{{[^}}]*nFaces\s+(\d+);[^}}]*startFace\s+(\d+);",
        text,
        re.S,
    )
    if not match:
        raise ValueError(f"Patch {patch_name} not found")
    return int(match.group(1)), int(match.group(2))


def patch_face_centres(points: np.ndarray, faces: list[list[int]], start_face: int, n_faces: int) -> np.ndarray:
    return np.array([points[face].mean(axis=0) for face in faces[start_face:start_face + n_faces]])


def parse_nonuniform_block(lines: list[str]) -> np.ndarray:
    count = None
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.isdigit():
            count = int(s)
            start = i + 2
            break
    if count is None or start is None:
        raise ValueError("Could not locate nonuniform list header")
    out = []
    for j in range(start, start + count):
        out.append([float(v) for v in lines[j].strip().strip("()").split()])
    return np.array(out)


def read_boundary_vector(path: Path, patch_name: str) -> np.ndarray:
    lines = path.read_text().splitlines()
    in_patch = False
    patch_depth = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s == patch_name:
            in_patch = True
            continue
        if not in_patch:
            continue
        if "{" in s:
            patch_depth += s.count("{")
        if "}" in s:
            patch_depth -= s.count("}")
            if patch_depth <= 0:
                break
        if s.startswith("value") and "nonuniform" in s:
            return parse_nonuniform_block(lines[i + 1:])
    raise ValueError(f"Could not read boundary values for patch {patch_name} from {path}")


def smooth_curve(x: np.ndarray, y: np.ndarray, nbins: int = 300) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    mids = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(x, bins) - 1
    idx = np.clip(idx, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        mask = idx == b
        if mask.any():
            xs.append(mids[b])
            ys.append(float(np.mean(y[mask])))
    return np.asarray(xs), np.asarray(ys)


def sign_change_crossings(x: np.ndarray, y: np.ndarray, eps: float = 1.0e-10) -> list[tuple[str, float]]:
    def sign(val: float) -> int:
        if val > eps:
            return 1
        if val < -eps:
            return -1
        return 0

    crossings = []
    prev_idx = None
    prev_sign = 0
    for i, yi in enumerate(y):
        curr_sign = sign(float(yi))
        if curr_sign == 0:
            continue
        if prev_idx is not None and curr_sign != prev_sign:
            x0 = float(x[prev_idx]); x1 = float(x[i])
            y0 = float(y[prev_idx]); y1 = float(y[i])
            frac = -y0 / (y1 - y0 + 1.0e-30)
            xc = x0 + frac * (x1 - x0)
            kind = "pos_to_neg" if prev_sign > 0 and curr_sign < 0 else "neg_to_pos"
            crossings.append((kind, xc))
        prev_idx = i
        prev_sign = curr_sign
    return crossings


def separation_metrics(x: np.ndarray, y: np.ndarray, x_min: float = 0.0) -> tuple[float | None, float | None]:
    mask = x >= x_min
    crossings = sign_change_crossings(x[mask], y[mask])
    onset = None
    reattach = None
    for kind, xc in crossings:
        if kind == "pos_to_neg" and onset is None:
            onset = xc
        elif kind == "neg_to_pos" and onset is not None and xc > onset:
            reattach = xc
            break
    return onset, reattach


def parse_model(case_dir: Path) -> str:
    text = read_text(case_dir / "constant" / "momentumTransport")
    m = re.search(r"\bmodel\s+([A-Za-z0-9_]+)\s*;", text)
    return m.group(1) if m else "RASModel"


def main() -> None:
    args = parse_args()
    case_dir = args.case.resolve()
    time_dir = (case_dir / args.time).resolve() if args.time else latest_time_with_field(case_dir, "wallShearStress")
    ref_path = (case_dir / args.reference).resolve() if not args.reference.is_absolute() else args.reference.resolve()
    out_dir = args.out.resolve() if args.out else case_dir / "comparison_exactmatch" / time_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.loadtxt(ref_path, delimiter=",", skiprows=1)
    x_ref = ref[:, 0]
    cf_ref = ref[:, 1]

    tp_text = read_text(case_dir / "constant" / "transportProperties")
    ub = parse_scalar_property(tp_text, "Ub")
    h = parse_scalar_property(tp_text, "h")
    nu = parse_scalar_property(tp_text, "nu")
    model_name = parse_model(case_dir)

    points = parse_points(case_dir / "constant" / "polyMesh" / "points")
    faces = parse_faces(case_dir / "constant" / "polyMesh" / "faces")
    n_faces, start_face = parse_patch_info(case_dir / "constant" / "polyMesh" / "boundary", "bottomWall")
    fc = patch_face_centres(points, faces, start_face, n_faces)
    wss = read_boundary_vector(time_dir / "wallShearStress", "bottomWall")
    x_case_raw = fc[:, 0] / h
    cf_case_raw = -2.0 * wss[:, 0] / (ub ** 2)
    x_case, cf_case = smooth_curve(x_case_raw, cf_case_raw, nbins=300)

    cf_interp = np.interp(x_ref, x_case, cf_case, left=np.nan, right=np.nan)
    mask = np.isfinite(cf_interp)
    rmse = float(np.sqrt(np.mean((cf_interp[mask] - cf_ref[mask]) ** 2)))
    sep_case, xr_case = separation_metrics(x_case, cf_case, x_min=0.0)
    sep_ref, xr_ref = separation_metrics(x_ref, cf_ref, x_min=0.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    ax.plot(x_ref, cf_ref, color="black", linewidth=2.2, label="Exact-match DNS reference")
    ax.plot(x_case, cf_case, color="#d62728", linewidth=1.9, label=f"{model_name} ({time_dir.name}) | RMSE={rmse:.6f} | x_r/h={xr_case:.4f}" if xr_case is not None else f"{model_name} ({time_dir.name}) | RMSE={rmse:.6f}")
    ax.axhline(0.0, color="0.55", linewidth=0.9)
    ax.set_xlabel("x/h")
    ax.set_ylabel("Cf")
    ax.set_title("Periodic-Hill Cf vs Exact-Match DNS Reference")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_dir / "01_cf_vs_exactmatch_reference.png", dpi=180)
    plt.close(fig)

    np.savetxt(out_dir / "cf_case.csv", np.column_stack([x_case, cf_case]), delimiter=",", header="x_over_h,Cf_case", comments="")
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Periodic-Hill Cf vs Exact-Match DNS Reference",
                "",
                f"- Case: `{case_dir}`",
                f"- Time used: `{time_dir.name}`",
                f"- Model: `{model_name}`",
                f"- Reference CSV: `{ref_path}`",
                f"- Reynolds number from case: `Re_h = {ub*h/nu:.1f}`",
                f"- Bulk velocity: `U_b = {ub}`",
                f"- Hill height: `h = {h}`",
                "",
                "## Error Metrics",
                "",
                f"- `C_f` RMSE vs exact-match DNS reference: `{rmse:.6f}`",
                "",
                "## Separation / Reattachment",
                "",
                f"- {model_name} separation onset estimate: `{sep_case}`",
                f"- Exact-match DNS separation onset estimate: `{sep_ref}`",
                f"- {model_name} reattachment estimate: `{xr_case}`",
                f"- Exact-match DNS reattachment estimate: `{xr_ref}`",
                "",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
