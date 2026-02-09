#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit(
        "plotly is required. If using conda, run: conda env create -f environment.yml"
    ) from exc

BOHR_TO_ANGSTROM = 0.529177210903

ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}

ELEMENT_COLORS = {
    1: "#eeeeee",  # H
    6: "#222222",  # C
    7: "#2d6cdf",  # N
    8: "#d1495b",  # O
}

ELEMENT_SIZES = {
    1: 8,
    6: 14,
    7: 14,
    8: 14,
}

# Keep sign encoding (positive vs negative) while still making each MO distinct in the legend.
POS_PALETTE = [
    "#1f77b4",
    "#2a6fbb",
    "#3b82f6",
    "#0ea5e9",
    "#22c1c3",
    "#2563eb",
    "#4f46e5",
    "#0f766e",
    "#0891b2",
    "#1d4ed8",
]
NEG_PALETTE = [
    "#d62728",
    "#e11d48",
    "#ef4444",
    "#f97316",
    "#fb7185",
    "#f43f5e",
    "#be123c",
    "#b91c1c",
    "#dc2626",
    "#ea580c",
]


def parse_orbital_index(comment: str):
    # Our exporter writes: "Orbital {idx} on NxNxN grid (padding ... Bohr)"
    parts = comment.strip().split()
    if len(parts) >= 2 and parts[0].lower() == "orbital":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_orbital_tag(comment: str):
    # "Orbital {idx} ({tag}) on ..."
    parts = comment.strip().split()
    if len(parts) >= 3 and parts[0].lower() == "orbital":
        token = parts[2]
        if token.startswith("(") and token.endswith(")"):
            return token[1:-1]
    return None


def read_cube(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        comment1 = handle.readline().rstrip("\n")
        comment2 = handle.readline().rstrip("\n")

        natoms_line = handle.readline().split()
        if len(natoms_line) < 4:
            raise ValueError(f"Invalid cube header in {path}")
        natoms = int(natoms_line[0])
        natoms_abs = abs(natoms)
        origin = np.array([float(natoms_line[1]), float(natoms_line[2]), float(natoms_line[3])])

        nx_line = handle.readline().split()
        ny_line = handle.readline().split()
        nz_line = handle.readline().split()
        nx = int(nx_line[0])
        ny = int(ny_line[0])
        nz = int(nz_line[0])
        ax = np.array([float(nx_line[1]), float(nx_line[2]), float(nx_line[3])])
        ay = np.array([float(ny_line[1]), float(ny_line[2]), float(ny_line[3])])
        az = np.array([float(nz_line[1]), float(nz_line[2]), float(nz_line[3])])

        atoms = []
        for _ in range(natoms_abs):
            parts = handle.readline().split()
            if len(parts) < 5:
                raise ValueError(f"Invalid atom line in {path}")
            atoms.append(
                {
                    "atomic_number": int(parts[0]),
                    "charge": float(parts[1]),
                    "position": np.array([float(parts[2]), float(parts[3]), float(parts[4])]),
                }
            )

        n_values = nx * ny * nz
        values = np.fromiter(
            (float(tok) for line in handle for tok in line.split()),
            dtype=float,
            count=n_values,
        )
        if values.size != n_values:
            raise ValueError(
                f"Cube {path} has {values.size} values, expected {n_values} (nx*ny*nz)"
            )

    orbital_index = parse_orbital_index(comment2)
    orbital_tag = parse_orbital_tag(comment2)

    return {
        "path": path,
        "comment1": comment1,
        "comment2": comment2,
        "orbital_index": orbital_index,
        "orbital_tag": orbital_tag,
        "origin": origin,
        "ax": ax,
        "ay": ay,
        "az": az,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "atoms": atoms,
        "values": values.reshape((nz, ny, nx)),  # matches z,y,x writing order
    }


def find_reference_atoms(benchmarks_path: Path, name: str):
    with benchmarks_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for entry in data.get("molecules", []):
        if entry.get("name", "").lower() == name.lower():
            atoms = []
            for atom in entry.get("atoms", []):
                atoms.append(
                    {
                        "atomic_number": int(atom["atomic_number"]),
                        "position": np.array(atom["position"], dtype=float),
                    }
                )
            return atoms
    raise ValueError(f"Reference molecule '{name}' not found in {benchmarks_path}")


def kabsch_align(source: np.ndarray, target: np.ndarray):
    if source.shape != target.shape:
        raise ValueError("source and target must have same shape")
    if source.shape[0] < 2:
        return np.eye(3), target.mean(axis=0) - source.mean(axis=0)

    c_src = source.mean(axis=0)
    c_tgt = target.mean(axis=0)
    src0 = source - c_src
    tgt0 = target - c_tgt

    h = src0.T @ tgt0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = c_tgt - (r @ c_src)
    return r, t


def bond_pairs(atoms):
    # Simple distance heuristic in Bohr (rough covalent radii in Angstrom converted to Bohr).
    radii_ang = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
    radii = {z: r / BOHR_TO_ANGSTROM for z, r in radii_ang.items()}

    pairs = []
    for i in range(len(atoms)):
        zi = atoms[i]["atomic_number"]
        ri = radii.get(zi, 0.8 / BOHR_TO_ANGSTROM)
        for j in range(i + 1, len(atoms)):
            zj = atoms[j]["atomic_number"]
            rj = radii.get(zj, 0.8 / BOHR_TO_ANGSTROM)
            cutoff = 1.3 * (ri + rj)
            dist = np.linalg.norm(atoms[i]["position"] - atoms[j]["position"])
            if dist <= cutoff:
                pairs.append((i, j))
    return pairs


def grid_coordinates(cube):
    # Our exporter writes orthogonal axes, but support the general case.
    nx, ny, nz = cube["nx"], cube["ny"], cube["nz"]
    origin = cube["origin"]
    ax, ay, az = cube["ax"], cube["ay"], cube["az"]

    xs = origin[0] + np.arange(nx) * ax[0]
    ys = origin[1] + np.arange(ny) * ay[1]
    zs = origin[2] + np.arange(nz) * az[2]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return xx, yy, zz


def orbital_center_of_mass(values, xx, yy, zz):
    w = values * values
    total = float(w.sum())
    if total <= 0.0:
        return None
    x = float((xx * w).sum() / total)
    y = float((yy * w).sum() / total)
    z = float((zz * w).sum() / total)
    return np.array([x, y, z])


def cube_bounds(cube):
    origin = cube["origin"]
    ax = cube["ax"] * (cube["nx"] - 1)
    ay = cube["ay"] * (cube["ny"] - 1)
    az = cube["az"] * (cube["nz"] - 1)
    corners = []
    for sx in (0.0, 1.0):
        for sy in (0.0, 1.0):
            for sz in (0.0, 1.0):
                corners.append(origin + sx * ax + sy * ay + sz * az)
    corners = np.stack(corners, axis=0)
    return corners.min(axis=0), corners.max(axis=0)


def axis_ranges(points: np.ndarray, padding_frac: float = 0.08):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pad = padding_frac * span
    return lo - pad, hi + pad


def auto_iso(values: np.ndarray, quantile: float, floor: float = 1e-6):
    abs_vals = np.abs(values.ravel())
    q = float(np.quantile(abs_vals, quantile))
    if not math.isfinite(q) or q <= 0.0:
        q = float(abs_vals.max())
    if not math.isfinite(q) or q <= 0.0:
        return None
    return max(q, floor)


def pick_nearest_index(origin: float, step: float, n: int, value: float):
    if n <= 1:
        return 0
    if step == 0.0:
        return int(n // 2)
    idx = int(round((value - origin) / step))
    return int(max(0, min(n - 1, idx)))


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D orbital visualization (Plotly).")
    parser.add_argument(
        "--cube", action="append", default=[], help="Input cube file (repeatable)."
    )
    parser.add_argument(
        "--cube-dir",
        default=None,
        help="Directory to load all *.cube files from (added after explicit --cube).",
    )
    parser.add_argument(
        "--cube-glob",
        default=None,
        help="Glob for cube files (e.g. 'out/h2o_orb*.cube' or 'out/**/*.cube').",
    )
    parser.add_argument("--out", default="orbitals.html", help="Output HTML path.")
    parser.add_argument(
        "--render",
        choices=["points", "slices", "isosurface"],
        default="points",
        help="How to render the volumetric data. 'points'/'slices' are best for debugging; 'isosurface' is best for blog-style visuals.",
    )
    parser.add_argument(
        "--field",
        choices=["psi", "abs", "density"],
        default="psi",
        help="Scalar field to plot from cube values: psi (signed), abs(|psi|), density(|psi|^2).",
    )
    parser.add_argument(
        "--normalize",
        choices=["orbital", "none"],
        default="orbital",
        help="Normalize each orbital field to [-1,1] or [0,1] before plotting (recommended for debugging).",
    )
    parser.add_argument(
        "--color-quantile",
        type=float,
        default=0.995,
        help="When --normalize none, use this quantile to set the shared color range.",
    )
    parser.add_argument(
        "--slice-opacity",
        type=float,
        default=0.75,
        help="Opacity of slice planes when --render slices.",
    )
    parser.add_argument(
        "--slice-threshold-frac",
        type=float,
        default=0.02,
        help="Mask slice pixels with values below this fraction of that orbital's max (reduces 'giant dark planes').",
    )
    parser.add_argument(
        "--slice-axes",
        default="xyz",
        help="Which orthogonal slices to render in --render slices (subset of 'xyz').",
    )
    parser.add_argument(
        "--slice-at",
        choices=["com", "center"],
        default="center",
        help="Where to place slice planes in --render slices (orbital COM or cube center).",
    )
    parser.add_argument(
        "--point-quantile",
        type=float,
        default=0.997,
        help="In --render points, keep only points above this quantile of the chosen field magnitude.",
    )
    parser.add_argument(
        "--point-max-points",
        type=int,
        default=8000,
        help="Max points per orbital (or per sign for signed fields).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Marker size for --render points.",
    )
    parser.add_argument(
        "--point-opacity",
        type=float,
        default=0.6,
        help="Marker opacity for --render points.",
    )
    parser.add_argument(
        "--point-seed",
        type=int,
        default=0,
        help="Random seed for downsampling in --render points (set to -1 for non-deterministic).",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=None,
        help="Isosurface value (in cube units). If omitted, uses --iso-quantile.",
    )
    parser.add_argument(
        "--iso-quantile",
        type=float,
        default=0.999,
        help="If --iso is omitted, pick iso as this quantile of |psi| over the grid.",
    )
    parser.add_argument(
        "--iso-min-frac",
        type=float,
        default=0.02,
        help="If auto iso is tiny, clamp it to this fraction of max(|psi|) to avoid huge/clipped surfaces.",
    )
    parser.add_argument(
        "--allow-clipping",
        action="store_true",
        help="Allow the chosen isovalue to touch the cube boundary (may look 'cut off').",
    )
    parser.add_argument(
        "--atoms-only",
        action="store_true",
        help="Only render atoms/bonds (ignore cube volumetric data).",
    )
    parser.add_argument(
        "--unit",
        choices=["bohr", "angstrom"],
        default="angstrom",
        help="Display unit.",
    )
    parser.add_argument("--reference", help="Optional benchmarks.json reference path.")
    parser.add_argument("--reference-name", help="Molecule name inside benchmarks.json.")
    parser.add_argument(
        "--show-cube-bounds",
        action="store_true",
        help="Draw the cube bounding box (useful to diagnose clipped isosurfaces).",
    )
    parser.add_argument(
        "--scale-bar",
        type=float,
        default=1.0,
        help="Draw a 3D scale bar of this length (in the chosen --unit). Set to 0 to disable.",
    )
    parser.add_argument(
        "--label-com",
        action="store_true",
        help="Render text labels next to orbital center-of-mass markers.",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(None if args.point_seed == -1 else args.point_seed)

    cube_paths = [Path(p) for p in args.cube]
    if args.cube_dir:
        cube_paths.extend(sorted(Path(args.cube_dir).glob("*.cube")))
    if args.cube_glob:
        cube_paths.extend(sorted(Path().glob(args.cube_glob)))
    # Preserve order but remove duplicates.
    seen = set()
    unique_paths = []
    for p in cube_paths:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        unique_paths.append(p)
    if not unique_paths:
        raise SystemExit("No cube files provided. Use --cube, --cube-dir, or --cube-glob.")

    cubes = [read_cube(p) for p in unique_paths]
    cubes.sort(
        key=lambda c: (
            c["orbital_index"] is None,
            c["orbital_index"] if c["orbital_index"] is not None else 10**9,
            str(c["path"]),
        )
    )
    unit_scale = BOHR_TO_ANGSTROM if args.unit == "angstrom" else 1.0

    fig = go.Figure()

    # If we're rendering distributions (slices), configure a shared color axis.
    coloraxis = None
    if not args.atoms_only and args.render == "slices":
        if args.field == "psi":
            coloraxis = dict(
                colorscale=[
                    [0.0, "#d1495b"],  # -1
                    [0.5, "#f8f8f8"],  #  0
                    [1.0, "#3b6fb6"],  # +1
                ],
                colorbar=dict(title="psi (normalized)" if args.normalize == "orbital" else "psi"),
            )
        elif args.field == "abs":
            coloraxis = dict(
                colorscale="Viridis",
                colorbar=dict(
                    title="|psi| (normalized)" if args.normalize == "orbital" else "|psi|"
                ),
            )
        else:
            coloraxis = dict(
                colorscale="Viridis",
                colorbar=dict(
                    title="|psi|^2 (normalized)" if args.normalize == "orbital" else "|psi|^2"
                ),
            )
        # Keep the colorbar out of the legend's way.
        coloraxis["colorbar"]["x"] = 1.03
        coloraxis["colorbar"]["y"] = 0.5
        coloraxis["colorbar"]["len"] = 0.85
        coloraxis["colorbar"]["thickness"] = 18

    # Predicted geometry from first cube.
    pred_atoms = cubes[0]["atoms"]
    pred_pos = np.stack([a["position"] for a in pred_atoms], axis=0)
    cube_mins = []
    cube_maxs = []
    for cube in cubes:
        cmin, cmax = cube_bounds(cube)
        cube_mins.append(cmin)
        cube_maxs.append(cmax)
    cube_min = np.stack(cube_mins, axis=0).min(axis=0)
    cube_max = np.stack(cube_maxs, axis=0).max(axis=0)

    # Draw cube bounds early so it doesn't spam the legend.
    if args.show_cube_bounds:
        cmin = cube_min * unit_scale
        cmax = cube_max * unit_scale
        corners = np.array(
            [
                [cmin[0], cmin[1], cmin[2]],
                [cmax[0], cmin[1], cmin[2]],
                [cmin[0], cmax[1], cmin[2]],
                [cmax[0], cmax[1], cmin[2]],
                [cmin[0], cmin[1], cmax[2]],
                [cmax[0], cmin[1], cmax[2]],
                [cmin[0], cmax[1], cmax[2]],
                [cmax[0], cmax[1], cmax[2]],
            ],
            dtype=float,
        )
        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        ex = []
        ey = []
        ez = []
        for a, b in edges:
            ex.extend([corners[a, 0], corners[b, 0], None])
            ey.extend([corners[a, 1], corners[b, 1], None])
            ez.extend([corners[a, 2], corners[b, 2], None])
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                name="Cube bounds",
                line=dict(color="rgba(0,0,0,0.25)", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Optional reference overlay (aligned to predicted).
    ref_pos_aligned = None
    if args.reference and args.reference_name:
        ref_atoms = find_reference_atoms(Path(args.reference), args.reference_name)
        ref_pos = np.stack([a["position"] for a in ref_atoms], axis=0)
        if ref_pos.shape[0] == pred_pos.shape[0]:
            r, t = kabsch_align(ref_pos, pred_pos)
            ref_pos_aligned = (r @ ref_pos.T).T + t
        else:
            ref_pos_aligned = ref_pos
        fig.add_trace(
            go.Scatter3d(
                x=ref_pos_aligned[:, 0] * unit_scale,
                y=ref_pos_aligned[:, 1] * unit_scale,
                z=ref_pos_aligned[:, 2] * unit_scale,
                mode="markers",
                name="Reference atoms",
                marker=dict(size=5, color="#111111", opacity=0.4),
            )
        )

    # Predicted atoms (group by element so the legend clearly shows colors).
    atoms_by_z = {}
    for idx, atom in enumerate(pred_atoms):
        znum = atom["atomic_number"]
        atoms_by_z.setdefault(znum, []).append((idx, atom))
    for znum in sorted(atoms_by_z.keys()):
        symbol = ELEMENT_SYMBOLS.get(znum, str(znum))
        points = atoms_by_z[znum]
        xs = []
        ys = []
        zs = []
        hover = []
        for idx, atom in points:
            pos = atom["position"] * unit_scale
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            hover.append(f"{symbol}{idx} (Z={znum})")
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name=f"{symbol} atoms",
                marker=dict(
                    size=ELEMENT_SIZES.get(znum, 10),
                    color=ELEMENT_COLORS.get(znum, "#666666"),
                    line=dict(color="#111111", width=1),
                ),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    # Fix axis ranges so toggling orbitals doesn't re-scale the view.
    points = [pred_pos, np.vstack([cube_min, cube_max])]
    if ref_pos_aligned is not None:
        points.append(ref_pos_aligned)
    all_points = np.vstack(points) * unit_scale
    lo, hi = axis_ranges(all_points, padding_frac=0.10)
    span = np.maximum(hi - lo, 1e-8)
    aspect = span / float(span.max())

    # Bonds.
    pairs = bond_pairs(pred_atoms)
    if pairs:
        bx = []
        by = []
        bz = []
        for i, j in pairs:
            a = pred_atoms[i]["position"] * unit_scale
            b = pred_atoms[j]["position"] * unit_scale
            bx.extend([a[0], b[0], None])
            by.extend([a[1], b[1], None])
            bz.extend([a[2], b[2], None])
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="lines",
                name="Bonds",
                line=dict(color="#333333", width=4),
                showlegend=False,
            )
        )

    # Phase/marker legend keys (so the user doesn't have to infer colors from the title).
    if args.field == "psi":
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name="+ phase (blue hues)",
                marker=dict(size=8, color="#3b6fb6"),
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name="- phase (red hues)",
                marker=dict(size=8, color="#d1495b"),
                showlegend=True,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            name="Orbital COM (|psi|^2 centroid)",
            marker=dict(size=8, color="#ffd166", line=dict(color="#111111", width=1)),
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Scale bar (3D, rotates with the scene).
    if args.scale_bar and args.scale_bar > 0.0:
        bar = float(args.scale_bar)
        p0 = np.array([lo[0], lo[1], lo[2]], dtype=float)
        p0 = p0 + 0.06 * span  # inset from the corner
        p1 = p0 + np.array([bar, 0.0, 0.0], dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                name="Scale bar",
                line=dict(color="#111111", width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p1[0]],
                y=[p1[1]],
                z=[p1[2]],
                mode="text",
                text=[f"{bar:g} {args.unit}"],
                showlegend=False,
                hoverinfo="skip",
                textfont=dict(size=12, color="#111111"),
            )
        )

    # Orbitals (each cube is one orbital).
    if not args.atoms_only:
        # Compute shared color range when we're not normalizing.
        if args.render == "slices" and args.normalize == "none" and coloraxis is not None:
            vmax = 0.0
            for cube in cubes:
                v = cube["values"]
                if args.field == "psi":
                    vals = np.abs(v).ravel()
                elif args.field == "abs":
                    vals = np.abs(v).ravel()
                else:
                    vals = (v * v).ravel()
                q = float(np.quantile(vals, args.color_quantile))
                if math.isfinite(q) and q > vmax:
                    vmax = q
            if not math.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            if args.field == "psi":
                coloraxis["cmin"] = -vmax
                coloraxis["cmax"] = vmax
            else:
                coloraxis["cmin"] = 0.0
                coloraxis["cmax"] = vmax
        elif args.render == "slices" and args.normalize == "orbital" and coloraxis is not None:
            if args.field == "psi":
                coloraxis["cmin"] = -1.0
                coloraxis["cmax"] = 1.0
            else:
                coloraxis["cmin"] = 0.0
                coloraxis["cmax"] = 1.0

        show_colorbar = True
        for cube_idx, cube in enumerate(cubes):
            stem = cube["path"].stem
            orbital_index = cube.get("orbital_index")
            orbital_tag = cube.get("orbital_tag")
            if orbital_index is None:
                label = stem
            elif orbital_tag:
                label = f"MO {orbital_index} ({orbital_tag})"
            else:
                label = f"MO {orbital_index}"
            values = cube["values"]
            xx, yy, zz = grid_coordinates(cube)
            com = orbital_center_of_mass(values, xx, yy, zz)

            group = f"mo{orbital_index}" if orbital_index is not None else stem
            if args.render == "isosurface":
                x_flat = (xx * unit_scale).ravel()
                y_flat = (yy * unit_scale).ravel()
                z_flat = (zz * unit_scale).ravel()
                v_flat = values.ravel()

                if args.iso is not None:
                    iso = float(args.iso)
                else:
                    iso = auto_iso(values, args.iso_quantile)
                    if iso is None:
                        continue
                    max_abs = float(np.abs(values).max())
                    if math.isfinite(max_abs) and max_abs > 0.0:
                        iso = max(iso, args.iso_min_frac * max_abs)

                abs_vals = np.abs(values)
                boundary_max = float(
                    max(
                        abs_vals[0, :, :].max(),
                        abs_vals[-1, :, :].max(),
                        abs_vals[:, 0, :].max(),
                        abs_vals[:, -1, :].max(),
                        abs_vals[:, :, 0].max(),
                        abs_vals[:, :, -1].max(),
                    )
                )
                clipped = boundary_max >= iso
                if clipped and not args.allow_clipping:
                    iso = boundary_max * 1.05 + 1e-12
                    clipped = False
                eps = iso * 0.05 + 1e-6

                color_key = orbital_index if orbital_index is not None else cube_idx
                pos_color = POS_PALETTE[color_key % len(POS_PALETTE)]
                neg_color = NEG_PALETTE[color_key % len(NEG_PALETTE)]

                # Positive phase.
                fig.add_trace(
                    go.Isosurface(
                        x=x_flat,
                        y=y_flat,
                        z=z_flat,
                        value=v_flat,
                        isomin=iso,
                        isomax=iso + eps,
                        surface_count=1,
                        opacity=0.35,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        colorscale=[[0.0, pos_color], [1.0, pos_color]],
                        showscale=False,
                        name=f"{label} (iso={iso:.3g})" + (" [CLIPPED]" if clipped else ""),
                        legendgroup=group,
                        showlegend=True,
                    )
                )

                # Negative phase.
                fig.add_trace(
                    go.Isosurface(
                        x=x_flat,
                        y=y_flat,
                        z=z_flat,
                        value=v_flat,
                        isomin=-iso - eps,
                        isomax=-iso,
                        surface_count=1,
                        opacity=0.35,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        colorscale=[[0.0, neg_color], [1.0, neg_color]],
                        showscale=False,
                        name=f"{label} (-)",
                        legendgroup=group,
                        showlegend=False,
                    )
                )
            else:
                # Distribution view: points or slice planes.
                origin = cube["origin"]
                ax = np.array(cube["ax"], dtype=float)
                ay = np.array(cube["ay"], dtype=float)
                az = np.array(cube["az"], dtype=float)
                nx, ny, nz = cube["nx"], cube["ny"], cube["nz"]

                field = values
                if args.field == "abs":
                    field = np.abs(field)
                elif args.field == "density":
                    field = field * field

                # Per-orbital normalization makes shapes comparable even when magnitudes differ wildly.
                if args.normalize == "orbital":
                    denom = (
                        float(np.max(np.abs(field)))
                        if args.field == "psi"
                        else float(np.max(field))
                    )
                    if math.isfinite(denom) and denom > 0.0:
                        field = field / denom

                if args.render == "points":
                    mag = np.abs(field) if args.field == "psi" else field
                    thr = float(np.quantile(mag.ravel(), args.point_quantile))
                    if not math.isfinite(thr) or thr <= 0.0:
                        thr = float(mag.max()) * 0.1

                    color_key = orbital_index if orbital_index is not None else cube_idx
                    pos_color = POS_PALETTE[color_key % len(POS_PALETTE)]
                    neg_color = NEG_PALETTE[color_key % len(NEG_PALETTE)]

                    def emit_points(mask, name, color, showlegend):
                        idxs = np.argwhere(mask)
                        if idxs.size == 0:
                            return
                        if idxs.shape[0] > args.point_max_points:
                            sel = rng.choice(
                                idxs.shape[0], size=args.point_max_points, replace=False
                            )
                            idxs = idxs[sel]
                        # idxs are (z,y,x) for our values layout.
                        izs = idxs[:, 0].astype(float)
                        iys = idxs[:, 1].astype(float)
                        ixs = idxs[:, 2].astype(float)
                        pts = (
                            np.array(origin, dtype=float)[None, :]
                            + ixs[:, None] * ax[None, :]
                            + iys[:, None] * ay[None, :]
                            + izs[:, None] * az[None, :]
                        )
                        pts = pts * unit_scale
                        fig.add_trace(
                            go.Scatter3d(
                                x=pts[:, 0],
                                y=pts[:, 1],
                                z=pts[:, 2],
                                mode="markers",
                                name=name,
                                legendgroup=group,
                                showlegend=showlegend,
                                marker=dict(
                                    size=float(args.point_size),
                                    color=color,
                                    opacity=float(args.point_opacity),
                                ),
                                hoverinfo="skip",
                            )
                        )

                    if args.field == "psi":
                        emit_points(field >= thr, f"{label} points", pos_color, True)
                        emit_points(field <= -thr, f"{label} points", neg_color, False)
                    else:
                        emit_points(field >= thr, f"{label} points", pos_color, True)
                else:
                    # Slice planes. These are full planes, so by default we mask low-value pixels
                    # to avoid rendering giant, nearly-black rectangles.
                    dx = float(np.linalg.norm(ax))
                    dy = float(np.linalg.norm(ay))
                    dz = float(np.linalg.norm(az))
                    xs = origin[0] + np.arange(nx) * ax[0]
                    ys = origin[1] + np.arange(ny) * ay[1]
                    zs = origin[2] + np.arange(nz) * az[2]

                    if args.slice_at == "com" and com is not None:
                        center = com
                    else:
                        cmin, cmax = cube_bounds(cube)
                        center = 0.5 * (cmin + cmax)

                    ix = pick_nearest_index(origin[0], ax[0], nx, float(center[0]))
                    iy = pick_nearest_index(origin[1], ay[1], ny, float(center[1]))
                    iz = pick_nearest_index(origin[2], az[2], nz, float(center[2]))

                    # Threshold as a fraction of max to hide near-zero pixels.
                    max_mag = float(np.max(np.abs(field))) if args.field == "psi" else float(np.max(field))
                    thr = float(args.slice_threshold_frac) * max_mag
                    if not math.isfinite(thr) or thr < 0.0:
                        thr = 0.0

                    axes = set(ch.lower() for ch in args.slice_axes if ch.lower() in ("x", "y", "z"))
                    show_legend = True

                    def mask_surface(xg, yg, zg, colors):
                        if thr <= 0.0:
                            return xg, yg, zg, colors
                        if args.field == "psi":
                            keep = np.abs(colors) >= thr
                        else:
                            keep = colors >= thr
                        xg = xg.copy()
                        yg = yg.copy()
                        zg = zg.copy()
                        colors = colors.copy()
                        xg[~keep] = np.nan
                        yg[~keep] = np.nan
                        zg[~keep] = np.nan
                        colors[~keep] = np.nan
                        return xg, yg, zg, colors

                    # x-slice (YZ plane).
                    if "x" in axes:
                        x0 = float(xs[ix] * unit_scale)
                        y_grid = (ys * unit_scale)[None, :].repeat(nz, axis=0)
                        z_grid = (zs * unit_scale)[:, None].repeat(ny, axis=1)
                        x_grid = np.full((nz, ny), x0, dtype=float)
                        colors = field[:, :, ix]
                        x_grid, y_grid, z_grid, colors = mask_surface(
                            x_grid, y_grid, z_grid, colors
                        )
                        fig.add_trace(
                            go.Surface(
                                x=x_grid,
                                y=y_grid,
                                z=z_grid,
                                surfacecolor=colors,
                                coloraxis="coloraxis",
                                opacity=float(args.slice_opacity),
                                showscale=show_colorbar,
                                name=f"{label} slices",
                                legendgroup=group,
                                showlegend=show_legend,
                                hovertemplate=(
                                    f"MO {orbital_index} x-slice<br>"
                                    f"x={x0:.3f} {args.unit}<br>"
                                    f"y=%{{y:.3f}} {args.unit}<br>"
                                    f"z=%{{z:.3f}} {args.unit}<br>"
                                    "val=%{surfacecolor:.3g}<extra></extra>"
                                ),
                            )
                        )
                        show_colorbar = False
                        show_legend = False

                    # y-slice (XZ plane).
                    if "y" in axes:
                        y0 = float(ys[iy] * unit_scale)
                        x_grid = (xs * unit_scale)[None, :].repeat(nz, axis=0)
                        z_grid = (zs * unit_scale)[:, None].repeat(nx, axis=1)
                        y_grid = np.full((nz, nx), y0, dtype=float)
                        colors = field[:, iy, :]
                        x_grid, y_grid, z_grid, colors = mask_surface(
                            x_grid, y_grid, z_grid, colors
                        )
                        fig.add_trace(
                            go.Surface(
                                x=x_grid,
                                y=y_grid,
                                z=z_grid,
                                surfacecolor=colors,
                                coloraxis="coloraxis",
                                opacity=float(args.slice_opacity),
                                showscale=show_colorbar,
                                name=f"{label} slices",
                                legendgroup=group,
                                showlegend=show_legend,
                                hovertemplate=(
                                    f"MO {orbital_index} y-slice<br>"
                                    f"y={y0:.3f} {args.unit}<br>"
                                    f"x=%{{x:.3f}} {args.unit}<br>"
                                    f"z=%{{z:.3f}} {args.unit}<br>"
                                    "val=%{surfacecolor:.3g}<extra></extra>"
                                ),
                            )
                        )
                        show_colorbar = False
                        show_legend = False

                    # z-slice (XY plane).
                    if "z" in axes:
                        z0 = float(zs[iz] * unit_scale)
                        x_grid = (xs * unit_scale)[None, :].repeat(ny, axis=0)
                        y_grid = (ys * unit_scale)[:, None].repeat(nx, axis=1)
                        z_grid = np.full((ny, nx), z0, dtype=float)
                        colors = field[iz, :, :]
                        x_grid, y_grid, z_grid, colors = mask_surface(
                            x_grid, y_grid, z_grid, colors
                        )
                        fig.add_trace(
                            go.Surface(
                                x=x_grid,
                                y=y_grid,
                                z=z_grid,
                                surfacecolor=colors,
                                coloraxis="coloraxis",
                                opacity=float(args.slice_opacity),
                                showscale=show_colorbar,
                                name=f"{label} slices",
                                legendgroup=group,
                                showlegend=show_legend,
                                hovertemplate=(
                                    f"MO {orbital_index} z-slice<br>"
                                    f"z={z0:.3f} {args.unit}<br>"
                                    f"x=%{{x:.3f}} {args.unit}<br>"
                                    f"y=%{{y:.3f}} {args.unit}<br>"
                                    "val=%{surfacecolor:.3g}<extra></extra>"
                                ),
                            )
                        )
                        show_colorbar = False
                        show_legend = False

            # Orbital center-of-mass marker.
            if com is not None:
                com_mode = "markers+text" if args.label_com else "markers"
                com_text = [f"MO {orbital_index} COM"] if args.label_com else None
                fig.add_trace(
                    go.Scatter3d(
                        x=[com[0] * unit_scale],
                        y=[com[1] * unit_scale],
                        z=[com[2] * unit_scale],
                        mode=com_mode,
                        text=com_text,
                        name=f"{stem} COM",
                        legendgroup=group,
                        showlegend=False,
                        marker=dict(
                            size=6, color="#ffd166", line=dict(color="#111111", width=1)
                        ),
                        hovertemplate=(
                            "COM<br>"
                            f"x=%{{x:.3f}} {args.unit}<br>"
                            f"y=%{{y:.3f}} {args.unit}<br>"
                            f"z=%{{z:.3f}} {args.unit}<extra></extra>"
                        ),
                )
            )

    subtitle = (
        f"Loaded {len(cubes)} cube(s). "
        f"Render={args.render}, field={args.field}, normalize={args.normalize}. "
    )
    if args.render == "points":
        subtitle += f"Points keep top q={args.point_quantile:g} (max {args.point_max_points} pts). "
    elif args.render == "slices":
        subtitle += f"Slices mask below {args.slice_threshold_frac:g} of max. "
    subtitle += "Click MO legend items to toggle orbitals (view stays fixed)."

    fig.update_layout(
        title=f"Molecule + Molecular Orbitals<br><sup>{subtitle}</sup>",
        legend=dict(
            title_text="Legend",
            groupclick="togglegroup",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant",
            tracegroupgap=6,
        ),
        uirevision="orbitals",
        scene=dict(
            xaxis=dict(title=f"x ({args.unit})", range=[float(lo[0]), float(hi[0])], autorange=False),
            yaxis=dict(title=f"y ({args.unit})", range=[float(lo[1]), float(hi[1])], autorange=False),
            zaxis=dict(title=f"z ({args.unit})", range=[float(lo[2]), float(hi[2])], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=float(aspect[0]), y=float(aspect[1]), z=float(aspect[2])),
        ),
        margin=dict(l=0, r=120, t=40, b=0),
    )
    if coloraxis is not None and args.render == "slices":
        fig.update_layout(coloraxis=coloraxis)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
