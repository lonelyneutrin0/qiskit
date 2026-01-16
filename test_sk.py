from qiskit.synthesis import SolovayKitaevSynthesis
from qiskit.circuit.library import HGate, TGate, TdgGate, SGate, SdgGate, UnitaryGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary

import numpy as np

h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
t_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
tdg_matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

print("Test 1: Standard gates [H, T, Tdg]")
sk_standard = SolovayKitaevSynthesis(
    basis_gates=[HGate(), TGate(), TdgGate()],
    depth=10
)

print("\nTest 2: Custom UnitaryGate gates")
custom_gates = [
    UnitaryGate(h_matrix, label='custom_H'),
    UnitaryGate(t_matrix, label='custom_T'),
    UnitaryGate(tdg_matrix, label='custom_Tdg'),
]
sk_custom = SolovayKitaevSynthesis(
    basis_gates=custom_gates,
    depth=10
)

print("\nTest 3: Mixed standard gates and UnitaryGates")
rx_pi_4 = np.array([
    [np.cos(np.pi/8), -1j * np.sin(np.pi/8)],
    [-1j * np.sin(np.pi/8), np.cos(np.pi/8)]
], dtype=complex)

mixed_gates = [
    HGate(),
    TGate(),
    TdgGate(),
    UnitaryGate(rx_pi_4, label='RX_pi4'),
]
sk_mixed = SolovayKitaevSynthesis(
    basis_gates=mixed_gates,
    depth=8
)

# Test 4: Fidelity comparison between standard and custom gatesets
print("\nTest 4: Fidelity comparison (N=100 random unitaries)")
N = 100
standard_fidelities = np.zeros(N)
custom_fidelities = np.zeros(N)
cross_fidelities = np.zeros(N)

for i in range(N):
    unitary_gate = UnitaryGate(random_unitary(dims=2))

    # Synthesize the random unitary
    sk_standard_circuit_data = sk_standard.synthesize(unitary_gate, recursion_degree=2)
    sk_custom_circuit_data = sk_custom.synthesize(unitary_gate, recursion_degree=2)

    sk_standard_circuit = QuantumCircuit._from_circuit_data(sk_standard_circuit_data)
    sk_custom_circuit = QuantumCircuit._from_circuit_data(sk_custom_circuit_data)

    op_standard = Operator(sk_standard_circuit)
    op_custom = Operator(sk_custom_circuit)
    op_target = Operator(unitary_gate)

    # Compute fidelities using trace formula
    fidelity_standard = np.abs(np.trace(op_target.data.conj().T @ op_standard.data)) / 2
    fidelity_custom = np.abs(np.trace(op_target.data.conj().T @ op_custom.data)) / 2
    fidelity_standard_custom = np.abs(np.trace(op_standard.data.conj().T @ op_custom.data)) / 2

    standard_fidelities[i] = fidelity_standard
    custom_fidelities[i] = fidelity_custom
    cross_fidelities[i] = fidelity_standard_custom

print(f"  Standard gates - Mean fidelity: {standard_fidelities.mean():.6f}, Min: {standard_fidelities.min():.6f}")
print(f"  Custom gates   - Mean fidelity: {custom_fidelities.mean():.6f}, Min: {custom_fidelities.min():.6f}")
print(f"  Cross-fidelity - Mean: {cross_fidelities.mean():.6f}, Min: {cross_fidelities.min():.6f}")

# Assertions to verify fidelities are close to 1
assert standard_fidelities.min() > 0.99, f"Standard fidelity too low: {standard_fidelities.min()}"
assert custom_fidelities.min() > 0.99, f"Custom fidelity too low: {custom_fidelities.min()}"
assert cross_fidelities.min() > 0.99, f"Cross-fidelity too low: {cross_fidelities.min()}"

print("\n✓ All tests passed! Both standard and custom gatesets achieve high fidelity.")

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].hist(standard_fidelities, bins=20, alpha=0.7, color='blue')
    ax[0].set_title('Standard Basis Gates')
    ax[0].set_xlabel('Fidelity')
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlim(0.98, 1.01)

    ax[1].hist(custom_fidelities, bins=20, alpha=0.7, color='orange')
    ax[1].set_title('Custom UnitaryGate Basis')
    ax[1].set_xlabel('Fidelity')
    ax[1].set_xlim(0.98, 1.01)

    ax[2].hist(cross_fidelities, bins=20, alpha=0.7, color='green')
    ax[2].set_title('Standard vs Custom')
    ax[2].set_xlabel('Cross-Fidelity')
    ax[2].set_xlim(0.98, 1.01)

    plt.tight_layout()
    plt.savefig('sk_fidelity_comparison.png', dpi=150)
    print("\nFidelity histogram saved to 'sk_fidelity_comparison.png'")
except ImportError:
    print("\nmatplotlib not available, skipping visualization")
