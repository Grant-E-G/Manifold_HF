#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

ANGSTROM_TO_BOHR = 1.8897259886

ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def load_targets(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def embed_geometry(smiles: str, seed: int, max_iters: int):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required to generate geometries.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles}")
    if max_iters > 0:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    return mol


def mol_coordinates(mol):
    conf = mol.GetConformer()
    coords = []
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
    return coords


def mol_atoms(mol):
    atoms = []
    for atom in mol.GetAtoms():
        entry = {
            "element": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
        }
        isotope = atom.GetIsotope()
        if isotope:
            entry["isotope"] = isotope
        atoms.append(entry)
    return atoms


def compute_bond_lengths(mol, coords_bohr):
    lengths = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        dist = distance(coords_bohr[i], coords_bohr[j])
        lengths.append({"atoms": [i, j], "value": dist})
    lengths.sort(key=lambda entry: (entry["atoms"][0], entry["atoms"][1]))
    return lengths


def compute_bond_angles(mol, coords_bohr):
    angles = []
    neighbors = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    for center, neighs in neighbors.items():
        neighs = sorted(set(neighs))
        for idx_a in range(len(neighs)):
            for idx_b in range(idx_a + 1, len(neighs)):
                i = neighs[idx_a]
                k = neighs[idx_b]
                ang = angle(coords_bohr[i], coords_bohr[center], coords_bohr[k])
                angles.append({"atoms": [i, center, k], "value": ang})
    angles.sort(key=lambda entry: (entry["atoms"][1], entry["atoms"][0], entry["atoms"][2]))
    return angles


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def angle(a, b, c):
    v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def compute_energy(atoms, coords_angstrom, charge: int, multiplicity: int):
    try:
        from pyscf import gto, scf
    except ImportError as exc:
        raise RuntimeError("PySCF is required to compute HF energies.") from exc

    spin = multiplicity - 1
    if spin != 0:
        raise ValueError("Only closed-shell (multiplicity=1) targets are supported.")
    atom_list = []
    for atom, pos in zip(atoms, coords_angstrom):
        symbol = ELEMENT_SYMBOLS.get(atom["atomic_number"])
        if symbol is None:
            raise ValueError(f"Unsupported atomic number: {atom['atomic_number']}")
        atom_list.append([symbol, pos])

    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = "sto-3g"
    mol.charge = charge
    mol.spin = spin
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    e_tot = mf.kernel()
    e_nuc = mol.energy_nuc()
    e_elec = e_tot - e_nuc
    return e_tot, e_elec, e_nuc


def main():
    parser = argparse.ArgumentParser(description="Generate HF/STO-3G benchmark data.")
    parser.add_argument(
        "--targets",
        default="data/benchmarks.targets.json",
        help="Path to benchmark target list.",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks.json",
        help="Output path for benchmark data.",
    )
    parser.add_argument("--seed", type=int, default=7, help="RDKit random seed.")
    parser.add_argument(
        "--uff-iters",
        type=int,
        default=200,
        help="Number of UFF optimization iterations (0 to skip).",
    )
    args = parser.parse_args()

    targets_path = Path(args.targets)
    output_path = Path(args.output)

    data = load_targets(targets_path)
    molecules = []

    try:
        import rdkit
        from rdkit.Chem import rdBase
        rdkit_version = rdBase.rdkitVersion
    except Exception:
        rdkit_version = "unknown"

    try:
        import pyscf
        pyscf_version = pyscf.__version__
    except Exception:
        pyscf_version = "unknown"

    for entry in data.get("molecules", []):
        name = entry["name"]
        smiles = entry["smiles"]
        charge = entry.get("charge", 0)
        multiplicity = entry.get("multiplicity", 1)
        tags = entry.get("tags", [])

        mol = embed_geometry(smiles, args.seed, args.uff_iters)
        coords_angstrom = mol_coordinates(mol)
        atoms = mol_atoms(mol)

        e_tot, e_elec, e_nuc = compute_energy(atoms, coords_angstrom, charge, multiplicity)

        coords_bohr = [
            [coord[0] * ANGSTROM_TO_BOHR, coord[1] * ANGSTROM_TO_BOHR, coord[2] * ANGSTROM_TO_BOHR]
            for coord in coords_angstrom
        ]
        bond_lengths = compute_bond_lengths(mol, coords_bohr)
        bond_angles = compute_bond_angles(mol, coords_bohr)

        atoms_out = []
        for atom, pos in zip(atoms, coords_bohr):
            atom_entry = {
                "element": atom["element"],
                "atomic_number": atom["atomic_number"],
                "position": [pos[0], pos[1], pos[2]],
            }
            if "isotope" in atom:
                atom_entry["isotope"] = atom["isotope"]
            atoms_out.append(atom_entry)

        molecules.append(
            {
                "name": name,
                "smiles": smiles,
                "charge": charge,
                "multiplicity": multiplicity,
                "tags": tags,
                "atoms": atoms_out,
                "reference": {
                    "energy_total": e_tot,
                    "energy_electronic": e_elec,
                    "energy_nuclear": e_nuc,
                },
                "geometry": {
                    "bond_lengths": bond_lengths,
                    "bond_angles": bond_angles,
                },
            }
        )

    output = {
        "version": data.get("version", 1),
        "units": {"length": "bohr", "energy": "hartree", "angle": "deg"},
        "source": {
            "generator": "scripts/generate_benchmarks.py",
            "method": "HF",
            "basis": "STO-3G",
            "geometry": "RDKit ETKDGv3 + UFF",
            "rdkit_version": rdkit_version,
            "pyscf_version": pyscf_version,
        },
        "molecules": molecules,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
