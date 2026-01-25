//! Molecular system representation
//!
//! This module defines the molecular structure in a functional style.

/// Represents an atom in 3D space
#[derive(Debug, Clone)]
pub struct Atom {
    /// Atomic number (number of protons)
    pub atomic_number: u32,
    /// Position in 3D space (x, y, z) in Bohr
    pub position: [f64; 3],
}

impl Atom {
    /// Creates a new atom
    pub fn new(atomic_number: u32, position: [f64; 3]) -> Self {
        Self {
            atomic_number,
            position,
        }
    }

    /// Returns the nuclear charge (same as atomic number for atoms)
    pub fn nuclear_charge(&self) -> f64 {
        self.atomic_number as f64
    }
}

/// Represents a molecule as a collection of atoms
#[derive(Debug, Clone)]
pub struct Molecule {
    /// List of atoms in the molecule
    pub atoms: Vec<Atom>,
    /// Total charge of the molecule
    pub charge: i32,
    /// Spin multiplicity (2S + 1)
    pub multiplicity: u32,
}

impl Molecule {
    /// Creates a new molecule
    pub fn new(atoms: Vec<Atom>, charge: i32, multiplicity: u32) -> Self {
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    /// Returns the total number of electrons
    pub fn num_electrons(&self) -> usize {
        let nuclear_charge: u32 = self.atoms.iter()
            .map(|a| a.atomic_number)
            .sum();
        
        (nuclear_charge as i32 - self.charge) as usize
    }

    /// Returns the number of occupied orbitals (closed shell)
    pub fn num_occupied(&self) -> usize {
        self.num_electrons() / 2
    }

    /// Computes nuclear repulsion energy
    pub fn nuclear_repulsion(&self) -> f64 {
        let n = self.atoms.len();
        let mut energy = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let atom_i = &self.atoms[i];
                let atom_j = &self.atoms[j];

                let dx = atom_i.position[0] - atom_j.position[0];
                let dy = atom_i.position[1] - atom_j.position[1];
                let dz = atom_i.position[2] - atom_j.position[2];

                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                
                energy += atom_i.nuclear_charge() * atom_j.nuclear_charge() / distance;
            }
        }

        energy
    }

    /// Creates a hydrogen molecule (H2) at standard bond length
    pub fn h2() -> Self {
        let bond_length = 1.4; // Bohr
        Self::new(
            vec![
                Atom::new(1, [0.0, 0.0, 0.0]),
                Atom::new(1, [0.0, 0.0, bond_length]),
            ],
            0,
            1,
        )
    }

    /// Creates a water molecule (H2O)
    pub fn h2o() -> Self {
        // Positions in Bohr, approximate geometry
        Self::new(
            vec![
                Atom::new(8, [0.0, 0.0, 0.0]),           // O
                Atom::new(1, [0.0, 1.43, 1.11]),         // H
                Atom::new(1, [0.0, -1.43, 1.11]),        // H
            ],
            0,
            1,
        )
    }

    /// Creates a heavy water molecule (D2O)
    /// Note: Deuterium has the same atomic number as hydrogen (1),
    /// so electronically it's identical to H2O in the Born-Oppenheimer approximation
    pub fn d2o() -> Self {
        // Same geometry as H2O (deuterium has same atomic number as hydrogen)
        Self::new(
            vec![
                Atom::new(8, [0.0, 0.0, 0.0]),           // O
                Atom::new(1, [0.0, 1.43, 1.11]),         // D (atomic number 1)
                Atom::new(1, [0.0, -1.43, 1.11]),        // D (atomic number 1)
            ],
            0,
            1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_h2_electrons() {
        let h2 = Molecule::h2();
        assert_eq!(h2.num_electrons(), 2);
        assert_eq!(h2.num_occupied(), 1);
    }

    #[test]
    fn test_h2_nuclear_repulsion() {
        let h2 = Molecule::h2();
        let e_nuc = h2.nuclear_repulsion();
        // For H2 at 1.4 Bohr: 1*1/1.4 ≈ 0.714
        assert_abs_diff_eq!(e_nuc, 0.714285714, epsilon = 1e-6);
    }

    #[test]
    fn test_h2o_electrons() {
        let h2o = Molecule::h2o();
        assert_eq!(h2o.num_electrons(), 10);
        assert_eq!(h2o.num_occupied(), 5);
    }

    #[test]
    fn test_d2o_electrons() {
        let d2o = Molecule::d2o();
        // D2O has same electronic structure as H2O
        assert_eq!(d2o.num_electrons(), 10);
        assert_eq!(d2o.num_occupied(), 5);
    }
}
