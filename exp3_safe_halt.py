
# -*- coding: utf-8 -*-
"""
Experiment 3: Safe HALT – Quantum‑like Destructive Interference
===============================================================
FINAL VERSION – READY FOR PUBLICATION

Demonstrates that contradictory rules (X→Y and X→¬Y) cancel out,
driving the extracted state norm to zero. At α = 0.5 (equal mix),
energy collapses below threshold → Safe HALT triggered.

All results saved in ./results/exp3/
Figure: safe_halt_collapse.pdf
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION
# ===============================
CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'D': 10000,                          # hyperdimensionality
    'alpha_values': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    'threshold': 0.05,                   # HALT if energy < threshold
    'results_dir': './results/exp3'
}
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(os.path.join(CONFIG['results_dir'], 'plots'), exist_ok=True)
device = CONFIG['device']
print(f"Device: {device}")
print(f"Results will be saved in: {CONFIG['results_dir']}\n")

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===============================
# HYPERDIMENSIONAL PRIMITIVES
# ===============================
def generate_phasor(dim, num_vectors=1):
    theta = (torch.rand(num_vectors, dim) * 2 * np.pi) - np.pi
    return torch.polar(torch.ones_like(theta), theta)

def bind(A, B):
    return A * B

def unbind(S, B):
    return S * torch.conj(B)

def shift(X):
    return torch.roll(X, shifts=1, dims=-1)

# ===============================
# MAIN
# ===============================
def main():
    print("=" * 70)
    print("Experiment 3: Safe HALT")
    print("=" * 70)

    seed_everything(42)   # deterministic for reproducibility

    # Generate cause X and two opposite effects Y and ¬Y
    X = generate_phasor(CONFIG['D'], 1).squeeze(0)
    Y = generate_phasor(CONFIG['D'], 1).squeeze(0)
    notY = -Y

    rule1 = bind(shift(X), Y)
    rule2 = bind(shift(X), notY)

    energies = []
    results = []

    for alpha in CONFIG['alpha_values']:
        # Superposition of contradictory rules
        G = alpha * rule1 + (1 - alpha) * rule2
        effect = unbind(G, shift(X))
        energy = torch.norm(effect).item() / np.sqrt(CONFIG['D'])
        halt = energy < CONFIG['threshold']
        # Convert to standard Python types for JSON serialization
        energy_float = float(energy)
        halt_bool = bool(halt)
        energies.append(energy_float)
        results.append({
            'alpha': float(alpha),
            'energy': energy_float,
            'halt': halt_bool
        })
        print(f"α={alpha:.1f}, energy={energy_float:.6f} → {'HALT' if halt_bool else 'OK'}")

    # Save numerical results
    with open(os.path.join(CONFIG['results_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Summary
    summary = {
        'threshold': float(CONFIG['threshold']),
        'alpha_collapse': 0.5,
        'energy_at_collapse': energies[CONFIG['alpha_values'].index(0.5)],
        'max_energy': float(max(energies)),
        'min_energy': float(min(energies))
    }
    with open(os.path.join(CONFIG['results_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(CONFIG['alpha_values'], energies, 'bo-', linewidth=2, markersize=8, label='State norm')
    plt.axhline(y=CONFIG['threshold'], color='r', linestyle='--', linewidth=2, label='HALT threshold')
    plt.fill_between(CONFIG['alpha_values'], 0, CONFIG['threshold'], alpha=0.3, color='red')
    plt.xlabel('α (weight of rule X→Y)', fontsize=12)
    plt.ylabel('Normalized state norm ||S|| / √D', fontsize=12)
    plt.title('Safe HALT: Destructive interference at α=0.5', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_dir'], 'plots', 'safe_halt_collapse.pdf'), dpi=150)
    plt.savefig(os.path.join(CONFIG['results_dir'], 'plots', 'safe_halt_collapse.png'), dpi=150)
    plt.close()

    print(f"\nPlot saved to {os.path.join(CONFIG['results_dir'], 'plots')}")
    print("✅ Safe HALT demonstration complete.")

if __name__ == "__main__":
    main()
