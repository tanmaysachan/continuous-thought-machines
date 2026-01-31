"""
Analyze Hebbian Weight Matrices from Checkpoints

Creates detailed visualizations of lateral connection weights:
- Zoomed weight matrix view (reveals structure hidden at full scale)
- Weight distribution histogram (log scale)
- Per-neuron incoming weight analysis (identifies "hub" neurons)

Usage:
    python scripts/analyze_hebbian_weights.py --checkpoint logs/hebbian/mazes-small-mac/checkpoint_46000.pt

    # Compare two checkpoints (e.g., standard vs Oja)
    python scripts/analyze_hebbian_weights.py \
        --checkpoint logs/hebbian/mazes-small-mac/checkpoint_46000.pt \
        --checkpoint2 logs/hebbian-oja/mazes-small-mac/checkpoint_41000.pt \
        --labels "Standard Hebbian" "Oja Rule"
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_hebbian_data(checkpoint_path):
    """Load Hebbian-related weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    data = {
        'base_lateral': None,
        'lateral_gate': None,
        'step': ckpt.get('iteration', 0),
        'args': ckpt.get('args', None),
    }

    for key, value in state_dict.items():
        if 'hebbian.base_lateral' in key:
            data['base_lateral'] = value.numpy()
        elif 'hebbian.lateral_gate_weights' in key:
            data['lateral_gate'] = value.numpy()

    return data


def analyze_single(checkpoint_path, output_path, zoom_size=100):
    """Analyze a single checkpoint."""
    data = load_hebbian_data(checkpoint_path)
    weights = data['base_lateral']
    gate_raw = data['lateral_gate']
    step = data['step']

    if weights is None:
        print(f"No base_lateral weights found in {checkpoint_path}")
        return

    np.fill_diagonal(weights, 0)
    gate_sigmoid = 1 / (1 + np.exp(-gate_raw)) if gate_raw is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Hebbian Weight Analysis - Step {step}', fontsize=14)

    # Row 1: Weight matrix analysis
    # 1a. Zoomed view
    zoom = weights[:zoom_size, :zoom_size]
    vmax = np.abs(zoom).max() or 1
    im = axes[0, 0].imshow(zoom, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title(f'Weight Matrix (neurons 0-{zoom_size})')
    axes[0, 0].set_xlabel('Neuron j')
    axes[0, 0].set_ylabel('Neuron i')
    plt.colorbar(im, ax=axes[0, 0])

    # 1b. Weight distribution
    flat = weights.flatten()
    nonzero = flat[flat != 0]
    axes[0, 1].hist(nonzero, bins=100, alpha=0.7, color='blue', density=True)
    axes[0, 1].set_yscale('log')
    axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f'Weight Distribution\nmean={flat.mean():.4f}, std={flat.std():.4f}')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Density (log)')

    # 1c. Per-neuron incoming weight
    incoming = np.abs(weights).sum(axis=1)
    sorted_incoming = np.sort(incoming)[::-1]
    axes[0, 2].bar(range(len(incoming)), sorted_incoming, width=1, alpha=0.7, color='steelblue')
    axes[0, 2].set_title(f'Incoming Lateral Weight per Neuron\nmax={incoming.max():.2f}, mean={incoming.mean():.2f}')
    axes[0, 2].set_xlabel('Neuron (sorted)')
    axes[0, 2].set_ylabel('Sum of |weights|')

    # Row 2: Gate and connection analysis
    # 2a. Gate distribution
    if gate_sigmoid is not None:
        axes[1, 0].hist(gate_sigmoid, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        above_05 = (gate_sigmoid > 0.5).mean() * 100
        axes[1, 0].set_title(f'Gate Values (sigmoid)\nmean={gate_sigmoid.mean():.3f}, {above_05:.1f}% > 0.5')
        axes[1, 0].set_xlabel('Gate Value')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xlim([0, 1])
    else:
        axes[1, 0].text(0.5, 0.5, 'No gate data', ha='center', va='center', transform=axes[1, 0].transAxes)

    # 2b. Outgoing weight distribution
    outgoing = np.abs(weights).sum(axis=0)
    sorted_outgoing = np.sort(outgoing)[::-1]
    axes[1, 1].bar(range(len(outgoing)), sorted_outgoing, width=1, alpha=0.7, color='coral')
    axes[1, 1].set_title(f'Outgoing Lateral Weight per Neuron\nmax={outgoing.max():.2f}, mean={outgoing.mean():.2f}')
    axes[1, 1].set_xlabel('Neuron (sorted)')
    axes[1, 1].set_ylabel('Sum of |weights|')

    # 2c. Effective lateral influence (gate * incoming)
    if gate_sigmoid is not None:
        effective_incoming = gate_sigmoid * incoming * 0.1  # with lateral_strength
        sorted_effective = np.sort(effective_incoming)[::-1]
        axes[1, 2].bar(range(len(effective_incoming)), sorted_effective, width=1, alpha=0.7, color='purple')
        axes[1, 2].set_title(f'Effective Lateral Influence\n(gate × incoming × 0.1)')
        axes[1, 2].set_xlabel('Neuron (sorted)')
        axes[1, 2].set_ylabel('Effective weight')
    else:
        axes[1, 2].text(0.5, 0.5, 'No gate data', ha='center', va='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary statistics
    print(f"\n=== Summary (Step {step}) ===")
    print(f"Weight matrix: {weights.shape}")
    print(f"  Range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Std: {weights.std():.4f}")
    print(f"  Non-zero (>0.01): {(np.abs(flat) > 0.01).mean()*100:.1f}%")
    print(f"Incoming weights per neuron: mean={incoming.mean():.2f}, max={incoming.max():.2f}")
    if gate_sigmoid is not None:
        print(f"Gates: mean={gate_sigmoid.mean():.3f}, {(gate_sigmoid > 0.5).mean()*100:.1f}% > 0.5")


def analyze_comparison(ckpt1_path, ckpt2_path, labels, output_path, zoom_size=100):
    """Compare two checkpoints side by side."""
    data1 = load_hebbian_data(ckpt1_path)
    data2 = load_hebbian_data(ckpt2_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (data, label) in enumerate([(data1, labels[0]), (data2, labels[1])]):
        weights = data['base_lateral']
        if weights is None:
            continue
        np.fill_diagonal(weights, 0)

        # Zoomed view
        zoom = weights[:zoom_size, :zoom_size]
        vmax = np.abs(zoom).max() or 1
        im = axes[row, 0].imshow(zoom, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f'{label}\nZoomed (neurons 0-{zoom_size})')
        axes[row, 0].set_xlabel('Neuron j')
        axes[row, 0].set_ylabel('Neuron i')
        plt.colorbar(im, ax=axes[row, 0])

        # Weight distribution
        flat = weights.flatten()
        nonzero = flat[flat != 0]
        axes[row, 1].hist(nonzero, bins=100, alpha=0.7, color='blue')
        axes[row, 1].set_yscale('log')
        axes[row, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[row, 1].set_title(f'Weight Distribution (log scale)\nmean={flat.mean():.4f}, std={flat.std():.4f}')
        axes[row, 1].set_xlabel('Weight Value')

        # Incoming weight per neuron
        incoming = np.abs(weights).sum(axis=1)
        axes[row, 2].bar(range(len(incoming)), np.sort(incoming)[::-1], width=1, alpha=0.7)
        axes[row, 2].set_title(f'Incoming Lateral Weight per Neuron\n(sorted, max={incoming.max():.2f})')
        axes[row, 2].set_xlabel('Neuron (sorted)')
        axes[row, 2].set_ylabel('Sum of |weights|')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Hebbian weight matrices')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--checkpoint2', type=str, default=None,
                        help='Optional second checkpoint for comparison')
    parser.add_argument('--labels', type=str, nargs=2, default=['Model 1', 'Model 2'],
                        help='Labels for comparison (two strings)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization')
    parser.add_argument('--zoom', type=int, default=100,
                        help='Size of zoomed weight matrix view')
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
        output_path = os.path.join(ckpt_dir, f'{ckpt_name}_analysis.png')

    # Run analysis
    if args.checkpoint2:
        analyze_comparison(args.checkpoint, args.checkpoint2, args.labels, output_path, args.zoom)
    else:
        analyze_single(args.checkpoint, output_path, args.zoom)


if __name__ == '__main__':
    main()
