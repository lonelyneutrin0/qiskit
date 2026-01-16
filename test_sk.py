from qiskit.synthesis import SolovayKitaevSynthesis
from qiskit.circuit.library import HGate, TGate, TdgGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import numpy as np
import qiskit

# need to build THT^dag

_ = QuantumCircuit(1)
_.append(TGate(), [0])
_.append(HGate(), [0])
_.append(TdgGate(), [0])

h_matrix = qiskit.quantum_info.Operator(_).data
t_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
tdg_matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

sk_standard = SolovayKitaevSynthesis(
    basis_gates=[HGate(), TGate(), TdgGate()],
    depth=10
)

sk_custom = SolovayKitaevSynthesis.from_matrices(
    basis_matrices=[h_matrix, t_matrix, tdg_matrix],
    depth=10
)

N = 10_000
standard_fidelities = np.zeros(N)
custom_fidelities = np.zeros(N)
cross_fidelities = np.zeros(N)

for i in range(N):
    unitary_gate = qiskit.circuit.library.UnitaryGate(
        qiskit.quantum_info.random_unitary(dims=2),
    )

    # Synthesize the random unitary directly
    sk_standard_circuit_data = sk_standard.synthesize(unitary_gate, recursion_degree=2)
    sk_custom_circuit_data = sk_custom.synthesize(unitary_gate, recursion_degree=2)

    sk_standard_circuit = QuantumCircuit._from_circuit_data(sk_standard_circuit_data)
    sk_custom_circuit = QuantumCircuit._from_circuit_data(sk_custom_circuit_data)

    op_standard = Operator(sk_standard_circuit)
    op_custom = Operator(sk_custom_circuit)
    op_target = Operator(unitary_gate)

    equiv_standard = op_standard.equiv(op_target)
    equiv_custom = op_custom.equiv(op_target)
    equiv_to_each_other = op_standard.equiv(op_custom)

    # Compute fidelities using trace formula
    fidelity_standard = np.abs(np.trace(op_target.data.conj().T @ op_standard.data)) / 2
    fidelity_custom = np.abs(np.trace(op_target.data.conj().T @ op_custom.data)) / 2
    fidelity_standard_custom = np.abs(np.trace(op_standard.data.conj().T @ op_custom.data)) / 2

    standard_fidelities[i] = fidelity_standard
    custom_fidelities[i] = fidelity_custom
    cross_fidelities[i] = fidelity_standard_custom

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

ax[0].hist(standard_fidelities, bins=50, alpha=0.7, label='Clifford Gateset', color='blue')
ax[0].set_title('Standard Basis Gates')
ax[0].set_xlabel('Fidelity')
ax[0].set_ylabel('Frequency')
ax[0].legend()
ax[0].grid(True)

ax[1].hist(custom_fidelities, bins=50, alpha=0.7, label='Conjugate Clifford Gateset', color='orange')
ax[1].set_title('Custom Matrix Gates')
ax[1].set_xlabel('Fidelity')
ax[1].set_ylabel('Frequency')
ax[1].legend()
ax[1].grid(True)

ax[2].hist(cross_fidelities, bins=50, alpha=0.7, label='Cross Fidelity (Standard vs Custom)', color='green')
ax[2].set_title('Standard vs Custom Implementations')
ax[2].set_xlabel('Fidelity')
ax[2].set_ylabel('Frequency')
ax[2].legend()
ax[2].grid(True)

plt.show()
