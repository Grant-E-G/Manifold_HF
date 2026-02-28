#!/usr/bin/env python3
"""Compare H2O orbital-exchange asymmetry against a PySCF gold standard.

This script:
1) Runs the local renderer in debug mode to get the currently reported
   orbital-exchange asymmetry metric.
2) Recomputes the same centroid/exchange metric from a PySCF RHF/STO-3G solve
   on the same in-repo H2O geometry.
3) Reports both values and a simple similarity verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pyscf import gto, scf


H2O_BOHR = np.array(
    [
        [0.0, 0.0, 0.0],  # O
        [0.0, 1.43, 1.11],  # H
        [0.0, -1.43, 1.11],  # H
    ],
    dtype=float,
)


def parse_metric_value(svg_text: str, label: str) -> str:
    pattern = re.compile(
        rf">{re.escape(label)}</text><text[^>]*>([^<]+)<",
        flags=re.IGNORECASE,
    )
    match = pattern.search(svg_text)
    if not match:
        raise ValueError(f"Could not find metric label '{label}' in SVG.")
    return match.group(1).replace("&gt;", ">").strip()


def parse_orbital_indices(value: str) -> List[int]:
    value = value.strip()
    if value.startswith("MOs "):
        return [int(part) for part in value[4:].split(",") if part.strip()]
    if value.startswith("MO "):
        return [int(value[3:].strip())]
    raise ValueError(f"Unsupported MO set format: {value!r}")


def run_local_debug_svg(svg_path: Path) -> str:
    cmd = [
        "cargo",
        "run",
        "--example",
        "render_molecule",
        "--",
        "h2o",
        "--debug-metrics",
        "--out",
        str(svg_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if proc.stdout:
        sys.stderr.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return svg_path.read_text(encoding="utf-8")


def project_yz(point: np.ndarray) -> np.ndarray:
    return np.array([point[1], point[2]], dtype=float)


def unproject_yz(point2: np.ndarray, offset: float = 0.0) -> np.ndarray:
    return np.array([offset, point2[0], point2[1]], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-16:
        return v * 0.0
    return v / n


def reflect_across_plane(v: np.ndarray, normal_unit: np.ndarray) -> np.ndarray:
    return v - 2.0 * np.dot(v, normal_unit) * normal_unit


def compute_pyscf_orbital_exchange_metric(orbital_indices: List[int]) -> Tuple[float, Tuple[int, int], List[int]]:
    mol = gto.M(
        atom=[
            ("O", tuple(H2O_BOHR[0])),
            ("H", tuple(H2O_BOHR[1])),
            ("H", tuple(H2O_BOHR[2])),
        ],
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    valid = [idx for idx in orbital_indices if 0 <= idx < mf.mo_coeff.shape[1]]
    if len(valid) < 2:
        raise ValueError("Need at least two orbital indices for exchange metric.")

    projected = np.array([project_yz(p) for p in H2O_BOHR], dtype=float)
    min_x, max_x = projected[:, 0].min(), projected[:, 0].max()
    min_y, max_y = projected[:, 1].min(), projected[:, 1].max()
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    padding = 2.0
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    grid = 84
    xs = np.linspace(min_x, max_x, grid)
    ys = np.linspace(min_y, max_y, grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([np.zeros(xx.size), xx.ravel(), yy.ravel()])
    ao = mol.eval_gto("GTOval_sph", coords)

    centroids: List[Tuple[int, np.ndarray]] = []
    for orbital_index in valid:
        psi = ao @ mf.mo_coeff[:, orbital_index]
        w = psi * psi
        s = w.sum()
        if s <= 1e-16:
            continue
        cx = float((coords[:, 1] * w).sum() / s)
        cy = float((coords[:, 2] * w).sum() / s)
        centroids.append((orbital_index, np.array([cx, cy], dtype=float)))
    if len(centroids) < 2:
        raise ValueError("Could not compute at least two non-zero orbital centroids.")

    oxygen = H2O_BOHR[0]
    h1 = H2O_BOHR[1]
    h2 = H2O_BOHR[2]
    r1 = h1 - oxygen
    r2 = h2 - oxygen
    bisector = r1 + r2
    plane_normal = np.cross(r1, r2)
    exchange_plane_normal = normalize(np.cross(bisector, plane_normal))

    oxygen_2d = project_yz(oxygen)
    h1_2d = project_yz(h1)
    h2_2d = project_yz(h2)

    ranked = []
    for idx, (_, c) in enumerate(centroids):
        d_o = np.linalg.norm(c - oxygen_2d)
        d_h = min(np.linalg.norm(c - h1_2d), np.linalg.norm(c - h2_2d))
        ranked.append((idx, float(d_o - d_h)))
    ranked.sort(key=lambda item: item[1], reverse=True)

    active = [centroids[idx] for idx, _ in ranked[:2]]
    if len(active) < 2:
        active = centroids
    if len(active) < 2:
        raise ValueError("Insufficient active centroids for comparison.")

    max_residual = 0.0
    worst_pair = (active[0][0], active[0][0])
    for orbital_i, centroid_i in active:
        point_i = unproject_yz(centroid_i, 0.0)
        reflected = oxygen + reflect_across_plane(point_i - oxygen, exchange_plane_normal)
        reflected_2d = project_yz(reflected)

        best_dist = float("inf")
        best_j = orbital_i
        for orbital_j, centroid_j in active:
            dist = float(np.linalg.norm(reflected_2d - centroid_j))
            if dist < best_dist:
                best_dist = dist
                best_j = orbital_j

        if best_dist > max_residual:
            max_residual = best_dist
            worst_pair = (orbital_i, best_j)

    return max_residual, worst_pair, [idx for idx, _ in active]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svg",
        default="/tmp/h2o_gold_compare.svg",
        help="Path for temporary debug SVG output.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=20.0,
        help="Max allowed ratio local/pyscf to call metrics 'similar'.",
    )
    args = parser.parse_args()

    svg_path = Path(args.svg)
    svg_text = run_local_debug_svg(svg_path)

    ours_str = parse_metric_value(svg_text, "Orb exch max residual (Bohr)")
    ours_worst = parse_metric_value(svg_text, "Orb exch worst pair")
    mo_set = parse_metric_value(svg_text, "MO set")
    orbital_indices = parse_orbital_indices(mo_set)
    ours = float(ours_str)

    pyscf_val, pyscf_pair, pyscf_active = compute_pyscf_orbital_exchange_metric(orbital_indices)

    denom = max(abs(pyscf_val), 1e-15)
    ratio = abs(ours) / denom
    similar = ratio <= args.max_ratio

    report = {
        "local_metric_label": "Orb exch max residual (Bohr)",
        "local_value": ours,
        "local_worst_pair": ours_worst,
        "orbital_indices": orbital_indices,
        "pyscf_value": pyscf_val,
        "pyscf_worst_pair": f"MO {pyscf_pair[0]} -> MO {pyscf_pair[1]}",
        "pyscf_active_orbitals": pyscf_active,
        "ratio_local_over_pyscf": ratio,
        "similar_under_ratio_threshold": similar,
        "ratio_threshold": args.max_ratio,
    }
    print(json.dumps(report, indent=2))

    return 0 if similar else 2


if __name__ == "__main__":
    raise SystemExit(main())
