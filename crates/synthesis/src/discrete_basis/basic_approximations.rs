// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use nalgebra::{Matrix2, Matrix3};
use num_complex::{Complex, ComplexFloat};
use num_traits::FloatConst;
use numpy::{Complex64, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use qiskit_circuit::{
    NoBlocks, Qubit,
    circuit_data::CircuitData,
    circuit_instruction::OperationFromPython,
    operations::{ArrayType, Operation, OperationRef, Param, StandardGate, UnitaryGate},
    packed_instruction::{PackedInstruction, PackedOperation},
};
use rstar::{Point, RTree};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::OnceLock;
use thiserror::Error;

use super::math;

#[derive(Error, Debug)]
pub enum DiscreteBasisError {
    #[error("Parameterized gates cannot be decomposed.")]
    ParameterizedGate,

    #[error("Cannot extract matrix from operation.")]
    NoMatrix,
}

impl From<DiscreteBasisError> for PyErr {
    fn from(value: DiscreteBasisError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

/// A discrete gate that can be either a standard gate or a custom gate defined by its matrix.
///
/// This allows the Solovay-Kitaev algorithm to work with both built-in gates and user-defined
/// custom gates while maintaining a clean interface.
#[derive(Clone, Debug)]
pub enum DiscreteGate {
    /// A standard Qiskit gate (H, T, Tdg, etc.)
    Standard(StandardGate),
    /// A custom gate defined by its U(2) matrix, SO(3) representation, phase, and optional name
    Custom {
        /// The U(2) matrix representation of the gate
        matrix_u2: Matrix2<Complex64>,
        /// The SO(3) representation (cached for efficiency)
        matrix_so3: Matrix3<f64>,
        /// The global phase
        phase: f64,
        /// Optional name for the gate (for display purposes)
        name: Option<String>,
        /// Index into the original basis set (for serialization)
        basis_index: usize,
    },
}

impl DiscreteGate {
    /// Create a new custom gate from a U(2) matrix.
    pub fn custom(matrix_u2: Matrix2<Complex64>, basis_index: usize, name: Option<String>) -> Self {
        let (matrix_so3, phase) = math::u2_to_so3(&matrix_u2);
        DiscreteGate::Custom {
            matrix_u2,
            matrix_so3,
            phase,
            name,
            basis_index,
        }
    }

    /// Create from a StandardGate.
    pub fn standard(gate: StandardGate) -> Self {
        DiscreteGate::Standard(gate)
    }

    /// Get the SO(3) matrix representation.
    pub fn to_so3(&self) -> Result<(Matrix3<f64>, f64), DiscreteBasisError> {
        match self {
            DiscreteGate::Standard(gate) => math::standard_gates_to_so3(gate, &[]),
            DiscreteGate::Custom { matrix_so3, phase, .. } => Ok((*matrix_so3, *phase)),
        }
    }

    /// Get the U(2) matrix representation.
    pub fn to_u2(&self) -> Result<Matrix2<Complex64>, DiscreteBasisError> {
        match self {
            DiscreteGate::Standard(gate) => math::standard_gates_to_u2(gate, &[]),
            DiscreteGate::Custom { matrix_u2, .. } => Ok(*matrix_u2),
        }
    }

    /// Get the inverse of this gate.
    pub fn inverse(&self) -> Option<DiscreteGate> {
        match self {
            DiscreteGate::Standard(gate) => {
                let (inv_gate, _) = gate.inverse(&[])?;
                Some(DiscreteGate::Standard(inv_gate))
            }
            DiscreteGate::Custom { matrix_u2, basis_index, name, .. } => {
                // The inverse of a unitary is its conjugate transpose
                let inv_matrix = matrix_u2.adjoint();
                let inv_name = name.as_ref().map(|n| format!("{}_dag", n));
                Some(DiscreteGate::custom(inv_matrix, *basis_index, inv_name))
            }
        }
    }

    /// Get the name of the gate.
    pub fn name(&self) -> &str {
        match self {
            DiscreteGate::Standard(gate) => gate.name(),
            DiscreteGate::Custom { name, .. } => {
                name.as_deref().unwrap_or("custom")
            }
        }
    }

    /// Check if two gates are inverses of each other.
    pub fn is_inverse_of(&self, other: &DiscreteGate) -> bool {
        match (self, other) {
            (DiscreteGate::Standard(g1), DiscreteGate::Standard(g2)) => {
                if let Some((inv, _)) = g1.inverse(&[]) {
                    inv == *g2
                } else {
                    false
                }
            }
            (DiscreteGate::Custom { matrix_u2: m1, .. }, DiscreteGate::Custom { matrix_u2: m2, .. }) => {
                // Check if m1 * m2 ≈ I
                let product = m1 * m2;
                let identity = Matrix2::identity();
                (product - identity).norm() < 1e-10
            }
            _ => false,
        }
    }
}

impl PartialEq for DiscreteGate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DiscreteGate::Standard(g1), DiscreteGate::Standard(g2)) => g1 == g2,
            (DiscreteGate::Custom { basis_index: i1, .. }, DiscreteGate::Custom { basis_index: i2, .. }) => {
                i1 == i2
            }
            _ => false,
        }
    }
}


/// A sequence of single qubit gates and their matrix.
///
/// Gates are stored in **circuit order**, not in matrix multiplication order. That means that
/// e.g. [H, T] corresponds to the matrix U = T @ H. The matrix is not stored as U(2), but in
/// a SO(3) representation, which discards the global phase.
#[pyclass]
#[derive(Clone, Debug)]
pub struct GateSequence {
    /// The sequence of discrete gates (can be standard or custom).
    pub gates: Vec<DiscreteGate>,
    /// The SO(3) representation of the sequence. Note that this is only equal to SU(2) up to a sign.
    pub matrix_so3: Matrix3<f64>,
    /// A global phase taking the U(2) representation of the sequence to SU(2).
    pub phase: f64,
}

/// A serializable version of the [GateSequence] used to store and retrieve [BasicApproximations].
#[derive(Serialize, Deserialize)]
struct SerializableGateSequence {
    /// For standard gates: stores the gate ID as u8
    /// For custom gates: stores u8::MAX as a marker
    gate_markers: Vec<u8>,
    /// Indices for custom gates into the basis set
    custom_indices: Vec<usize>,
    matrix_so3: Vec<f64>,
    phase: f64,
}

impl From<&GateSequence> for SerializableGateSequence {
    fn from(value: &GateSequence) -> Self {
        let mut gate_markers = Vec::with_capacity(value.gates.len());
        let mut custom_indices = Vec::new();

        for gate in &value.gates {
            match gate {
                DiscreteGate::Standard(sg) => {
                    gate_markers.push(*sg as u8);
                }
                DiscreteGate::Custom { basis_index, .. } => {
                    gate_markers.push(u8::MAX); // Marker for custom gate
                    custom_indices.push(*basis_index);
                }
            }
        }

        // store the SO(3) matrix as flattened vector
        let matrix_so3 = value.matrix_so3.iter().copied().collect::<Vec<f64>>();

        Self {
            gate_markers,
            custom_indices,
            matrix_so3,
            phase: value.phase,
        }
    }
}

/// Context needed to deserialize a GateSequence (contains custom gate definitions)
pub struct DeserializationContext {
    pub custom_gates: Vec<DiscreteGate>,
}

impl GateSequence {
    /// Deserialize from a SerializableGateSequence with context for custom gates.
    pub fn from_serializable(value: &SerializableGateSequence, ctx: Option<&DeserializationContext>) -> Self {
        let mut gates = Vec::with_capacity(value.gate_markers.len());
        let mut custom_idx = 0;

        for &marker in &value.gate_markers {
            if marker == u8::MAX {
                // Custom gate - look up from context
                if let Some(context) = ctx {
                    let basis_index = value.custom_indices[custom_idx];
                    if basis_index < context.custom_gates.len() {
                        gates.push(context.custom_gates[basis_index].clone());
                    }
                }
                custom_idx += 1;
            } else {
                // Standard gate
                let gate = ::bytemuck::checked::cast::<_, StandardGate>(marker);
                gates.push(DiscreteGate::Standard(gate));
            }
        }

        let matrix_so3 = Matrix3::from_iterator(value.matrix_so3.clone());

        Self {
            gates,
            matrix_so3,
            phase: value.phase,
        }
    }

    /// Create a new, empty sequence.
    pub fn new() -> Self {
        Self {
            gates: vec![],
            matrix_so3: Matrix3::identity(),
            phase: 0.,
        }
    }

    /// Get the gate labels.
    pub fn label(&self) -> String {
        self.gates.iter().map(|gate| gate.name()).collect()
    }

    /// Merge two [GateSequence]s.
    ///
    /// ``self.dot(other)`` results in a sequence where the gates are ``other.gates + self.gates``.
    pub fn dot(&self, other: &GateSequence) -> GateSequence {
        // merge the gates
        let mut gates = Vec::with_capacity(other.gates.len() + self.gates.len());
        gates.extend_from_slice(&other.gates);
        gates.extend_from_slice(&self.gates);

        // update the matrices and global phase
        let phase = self.phase + other.phase; // map to [0, 2pi)
        let matrix_so3 = self.matrix_so3 * other.matrix_so3;

        Self {
            gates,
            matrix_so3,
            phase,
        }
    }

    /// Return the adjoint.
    pub fn adjoint(&self) -> GateSequence {
        // Flip the gate order and invert them
        let gates = self
            .gates
            .iter()
            .rev()
            .filter_map(|gate| gate.inverse())
            .collect();

        // The transpose of an orthogonal matrix is equal to its inverse
        let matrix_so3 = self.matrix_so3.transpose();

        Self {
            gates,
            matrix_so3,
            phase: -self.phase,
        }
    }

    /// Remove gate-inverse pairs in-place.
    pub fn inverse_cancellation(&mut self) {
        if self.gates.len() < 2 {
            return; // there is nothing to cancel for 0 or 1 gate(s)
        }

        let mut reduced_gates: Vec<DiscreteGate> = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            // if we have a last gate check whether it cancels with the next one
            if let Some(last) = reduced_gates.last() {
                if gate.is_inverse_of(last) {
                    reduced_gates.pop();
                    continue;
                }
            }
            // we didn't have a gate in the queue yet or we don't have a cancelling pair
            reduced_gates.push(gate.clone());
        }

        self.gates = reduced_gates;
    }

    /// Get the U(2) matrix implemented by the gates.
    pub fn u2(&self) -> Result<Matrix2<Complex<f64>>, DiscreteBasisError> {
        let mut out = Matrix2::identity();
        for gate in &self.gates {
            let matrix = gate.to_u2()?;
            out = matrix * out;
        }
        Ok(out)
    }

    /// Compute the phase the sequence needs to match the target sequence.
    ///
    /// This assumes that [self] is a good approximation to ``target``, otherwise the result
    /// may not make sense.
    pub fn compute_phase(
        &self,
        target_u2: &Matrix2<Complex64>,
        target_phase: f64,
    ) -> Result<f64, DiscreteBasisError> {
        let self_u2 = self.u2()?;
        let (target_first, self_first) = target_u2
            .iter()
            .zip(self_u2.iter())
            .find(|&(&el, _)| el.abs() >= 1. / 2.)
            .expect("At least one element in the unitary must be >= 1/2.");

        // When we convert SU(2) to SO(3) we lose sign information, which translates to a
        // global phase uncertainty of +-1 = exp(i pi). We fix this here by checking which phase
        // is a better match (one should be clearly correct if the algorithm converged).
        let phase_candidate = self.phase - target_phase;
        let coeff_candidate = Complex::new(0., phase_candidate).exp();
        let candidate = self_first * coeff_candidate;

        if (target_first - candidate).abs() < (target_first + candidate).abs() {
            // phase candidate is correct
            Ok(phase_candidate)
        } else {
            // off by a -1 sign, so shift by PI
            Ok(phase_candidate + f64::PI())
        }
    }

    /// Convert the sequence to a circuit.
    ///
    /// If a target sequence is given, match the phase of [self] to the target.
    pub fn to_circuit(
        &self,
        target: Option<(&Matrix2<Complex64>, f64)>,
    ) -> Result<CircuitData, DiscreteBasisError> {
        let global_phase = match target {
            Some((target_u2, target_phase)) => {
                Param::Float(self.compute_phase(target_u2, target_phase)?)
            }
            None => Param::Float(self.phase),
        };

        let mut circuit = CircuitData::with_capacity(1, 0, self.gates.len(), global_phase).unwrap();
        for gate in &self.gates {
            match gate {
                DiscreteGate::Standard(sg) => {
                    circuit.push_standard_gate(*sg, &[], &[Qubit(0)]).unwrap();
                }
                DiscreteGate::Custom { matrix_u2, name, .. } => {
                    // Create a UnitaryGate for custom gates, using the stored name as label
                    let packed_inst = PackedInstruction {
                        op: PackedOperation::from_unitary(Box::new(UnitaryGate {
                            array: ArrayType::OneQ(*matrix_u2),
                        })),
                        qubits: circuit.add_qargs(&[Qubit(0)]),
                        clbits: Default::default(),
                        params: None,
                        label: name.clone().map(|s| Box::new(s)),
                        #[cfg(feature = "cache_pygates")]
                        py_op: OnceLock::new(),
                    };
                    circuit.push(packed_inst).unwrap();
                }
            }
        }
        Ok(circuit)
    }

    /// Push a new discrete gate onto [self].
    pub fn push_discrete(&mut self, gate: DiscreteGate) -> Result<(), DiscreteBasisError> {
        let (so3_matrix, phase) = gate.to_so3()?;

        // update matrix representations and keep track of the gate
        self.matrix_so3 = so3_matrix * self.matrix_so3;
        self.phase += phase;
        self.gates.push(gate);

        Ok(())
    }

    /// Push a new standard gate onto [self].
    fn push(&mut self, gate: StandardGate) -> Result<(), DiscreteBasisError> {
        self.push_discrete(DiscreteGate::Standard(gate))
    }

    /// Return an iterator that adds every gate in ``additions`` to the current sequence.
    fn iter_additions<'a>(
        &'a self,
        additions: &'a [StandardGate],
    ) -> impl Iterator<Item = Result<GateSequence, DiscreteBasisError>> + 'a {
        additions.iter().map(|gate| {
            let mut out = self.clone();
            out.push(*gate)?;
            Ok(out)
        })
    }

    /// Return an iterator that adds every discrete gate in ``additions`` to the current sequence.
    pub fn iter_discrete_additions<'a>(
        &'a self,
        additions: &'a [DiscreteGate],
    ) -> impl Iterator<Item = Result<GateSequence, DiscreteBasisError>> + 'a {
        additions.iter().map(|gate| {
            let mut out = self.clone();
            out.push_discrete(gate.clone())?;
            Ok(out)
        })
    }
}

#[pymethods]
impl GateSequence {
    /// Initialize from a vector of standard gates, plus the SO(3) matrix.
    ///
    /// Legacy method for backward compatibility with Python SK.
    #[staticmethod]
    fn from_gates_and_matrix(
        gates: Vec<OperationFromPython<NoBlocks>>,
        matrix_so3: PyReadonlyArray2<f64>,
        phase: f64,
    ) -> PyResult<Self> {
        // extract the StandardGate from the input
        let gates = gates
            .iter()
            .map(|op| match op.operation.view() {
                OperationRef::StandardGate(gate) => Ok(DiscreteGate::Standard(gate)),
                _ => Err(PyValueError::new_err(
                    "Only standard gates are allowed in GateSequence.from_gates_and_matrix",
                )),
            })
            .collect::<PyResult<_>>()?;

        let matrix_so3 = matrix3_from_pyreadonly(&matrix_so3);
        Ok(Self {
            gates,
            matrix_so3,
            phase,
        })
    }
}

/// A point in the R* tree. Contains the SO(3) representation of the gate sequence, plus an
/// optional index to retrieve the gate sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct BasicPoint {
    point: [f64; 9],      // SO(3) representation
    index: Option<usize>, // index to a gate sequence -- could explore using GateSequence directly
}

impl BasicPoint {
    pub fn from_sequence(sequence: &GateSequence, index: usize) -> Self {
        Self {
            point: ::core::array::from_fn(|i| sequence.matrix_so3[(i % 3, i / 3)]),
            index: Some(index),
        }
    }

    pub fn from_matrix(matrix: &Matrix3<f64>) -> Self {
        Self {
            point: ::core::array::from_fn(|i| matrix[(i % 3, i / 3)]),
            index: None,
        }
    }
}

impl Point for BasicPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 9;

    fn generate(generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        BasicPoint {
            point: ::core::array::from_fn(generator),
            index: None, // this point has no associated index
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        self.point[index]
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        &mut self.point[index]
    }
}

/// The basic approximations for Solovay Kitaev.
///
/// This struct allows to construct a tree of basic approximations and to query the closest
/// sequence given an target sequence (or SO(3) matrix).
#[derive(Debug)]
pub struct BasicApproximations {
    /// All points as flattened SO(3) matrix stored in a R* tree. This does not include the
    /// sequence of gates, see ``approximations``.
    pub points: RTree<BasicPoint>,
    /// A map relating the indices in the R* tree to a sequence of gates. This allows to
    /// retrieve the gates implementing a SO(3) "point" in the tree.
    pub approximations: HashMap<usize, GateSequence>,
}

impl BasicApproximations {
    /// Generate a tree of basic approximations from a set of discrete standard gates and a
    /// maximum depth.
    ///
    /// This will compute an SO(3) representation of any sequence of gates in ``basis_gates`` of
    /// length up to ``depth`` and store it in a tree structure, if there is no other sequence
    /// within a radius of ``sqrt(tol)``.
    ///
    /// All gates must be single-qubit, discrete (i.e. take no parameter) gates.
    ///
    /// # Args
    ///
    /// - ``basis_gates`` - A slice of [StandardGate]s to use in basic approximation.
    /// - ``depth`` - The maximum gate depth of the basic approximations.
    /// - ``tol`` - Control the granularity of the tree; new sequences are accepted if they
    ///   are further than ``sqrt(tol)`` from an existing element.
    pub fn generate_from(
        basis_gates: &[StandardGate],
        depth: usize,
        tol: Option<f64>,
    ) -> Result<Self, DiscreteBasisError> {
        let mut points: RTree<BasicPoint> = RTree::new();
        let mut approximations: HashMap<usize, GateSequence> = HashMap::new();

        // identity approximation
        let root = GateSequence::new();
        points.insert(BasicPoint::from_sequence(&root, 0));
        approximations.insert(0, root);
        let mut index = 1;

        let mut this_level: Vec<GateSequence> = vec![GateSequence::new()];
        let mut next_level: Vec<GateSequence> = Vec::new();
        let radius_sq = tol.unwrap_or(1e-12);

        for _ in 0..depth {
            for node in this_level.iter() {
                for candidate in node.iter_additions(basis_gates) {
                    let candidate = candidate?;
                    let point = BasicPoint::from_sequence(&candidate, index);
                    if points
                        .locate_within_distance(point.clone(), radius_sq)
                        .next()
                        .is_none()
                    {
                        // we don't have this point yet
                        points.insert(point);
                        approximations.insert(index, candidate.clone());
                        index += 1;
                        next_level.push(candidate);
                    }
                }
            }
            this_level.clone_from(&next_level);
            next_level.clear();
        }

        Ok(Self {
            points,
            approximations,
        })
    }

    /// Generate a tree of basic approximations from a set of discrete gates (can be mixed
    /// standard and custom gates).
    ///
    /// This is the unified method that works with both StandardGates and custom gates.
    pub fn generate_from_discrete(
        basis_gates: &[DiscreteGate],
        depth: usize,
        tol: Option<f64>,
    ) -> Result<Self, DiscreteBasisError> {
        let mut points: RTree<BasicPoint> = RTree::new();
        let mut approximations: HashMap<usize, GateSequence> = HashMap::new();

        // identity approximation
        let root = GateSequence::new();
        points.insert(BasicPoint::from_sequence(&root, 0));
        approximations.insert(0, root);
        let mut index = 1;

        let mut this_level: Vec<GateSequence> = vec![GateSequence::new()];
        let mut next_level: Vec<GateSequence> = Vec::new();
        let radius_sq = tol.unwrap_or(1e-12);

        for _ in 0..depth {
            for node in this_level.iter() {
                for candidate in node.iter_discrete_additions(basis_gates) {
                    let candidate = candidate?;
                    let point = BasicPoint::from_sequence(&candidate, index);
                    if points
                        .locate_within_distance(point.clone(), radius_sq)
                        .next()
                        .is_none()
                    {
                        // we don't have this point yet
                        points.insert(point);
                        approximations.insert(index, candidate.clone());
                        index += 1;
                        next_level.push(candidate);
                    }
                }
            }
            this_level.clone_from(&next_level);
            next_level.clear();
        }

        Ok(Self {
            points,
            approximations,
        })
    }

    /// Generate a tree of basic approximations from a list of U(2) matrices.
    ///
    /// Each matrix is treated as a custom gate and will be properly tracked in the output
    /// sequences, allowing circuit generation to work correctly.
    pub fn generate_from_matrices(
        basis_matrices: &[Matrix2<Complex<f64>>],
        depth: usize,
        tol: Option<f64>,
    ) -> Result<Self, DiscreteBasisError> {
        // Convert matrices to DiscreteGate::Custom
        let basis_gates: Vec<DiscreteGate> = basis_matrices
            .iter()
            .enumerate()
            .map(|(idx, m)| DiscreteGate::custom(*m, idx, Some(format!("custom_{}", idx))))
            .collect();

        Self::generate_from_discrete(&basis_gates, depth, tol)
    }

    /// Load from a slice of [GateSequence] objects.
    ///
    /// This is for legacy compatibility with the old Python version of SK.
    pub fn load_from_sequences(sequences: &[GateSequence]) -> Self {
        let mut points: RTree<BasicPoint> = RTree::new();
        let mut approximations: HashMap<usize, GateSequence> = HashMap::new();
        for (unique_index, sequence) in sequences.iter().enumerate() {
            approximations.insert(unique_index, sequence.clone());
            points.insert(BasicPoint::from_sequence(sequence, unique_index));
        }
        Self {
            points,
            approximations,
        }
    }

    /// Query the closest point to a [GateSequence].
    pub fn query(&self, matrix: &Matrix3<f64>) -> Option<&GateSequence> {
        let query_point = BasicPoint::from_matrix(matrix);
        self.points.nearest_neighbor(&query_point).map(|point| {
            let index = point
                .index
                .expect("All registered sequences should have an index. Blame a dev.");
            self.approximations
                .get(&index)
                .expect("All available indices should have a sequence. Also blame a dev.")
        })
    }

    /// Save the basic approximations into a file. This can be used to load the object again,
    /// see [Self::load].
    ///
    /// Note: When saving sequences with custom gates, the custom gate matrices are not saved.
    /// Use `load_with_context` to restore them.
    pub fn save(&self, filename: &str) -> ::std::io::Result<()> {
        // we turn the HashMap with GateSequences as keys into a HashMap
        // with SerializableGateSequence as key
        let serializable_approx = self
            .approximations
            .iter()
            .map(|(key, value)| (*key, SerializableGateSequence::from(value)))
            .collect::<HashMap<usize, SerializableGateSequence>>();

        // store the now serializable HashMap
        let file = ::std::fs::File::create(filename)?;
        bincode::serialize_into(file, &serializable_approx).map_err(::std::io::Error::other)?;
        Ok(())
    }

    /// Load the basic approximations from a file. See [Self::save] for saving the object.
    ///
    /// Note: This only works correctly for sequences containing only StandardGates.
    /// For sequences with custom gates, use `load_with_context`.
    pub fn load(filename: &str) -> ::std::io::Result<Self> {
        Self::load_with_context(filename, None)
    }

    /// Load the basic approximations from a file with context for custom gates.
    ///
    /// # Arguments
    /// * `filename` - Path to the saved file
    /// * `ctx` - Optional context containing custom gate definitions for deserialization
    pub fn load_with_context(filename: &str, ctx: Option<&DeserializationContext>) -> ::std::io::Result<Self> {
        let file = ::std::fs::File::open(filename)?;
        let serializable_approx: HashMap<usize, SerializableGateSequence> =
            bincode::deserialize_from(file).map_err(::std::io::Error::other)?;

        // construct the GateSequence from its serializable version
        let approximations = serializable_approx
            .iter()
            .map(|(key, value)| (*key, GateSequence::from_serializable(value, ctx)))
            .collect::<HashMap<usize, GateSequence>>();

        // build the RTree from the sequences
        let mut points: RTree<BasicPoint> = RTree::new();
        for (index, sequence) in approximations.iter() {
            points.insert(BasicPoint::from_sequence(sequence, *index));
        }

        Ok(Self {
            points,
            approximations,
        })
    }
}

#[inline]
fn matrix3_from_pyreadonly(array: &PyReadonlyArray2<f64>) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| *array.get((i, j)).unwrap())
}
