"""
Visualize Hebbian Weight Matrix Evolution

Loads checkpoints and creates GIFs showing how the lateral connection weights
evolved during training.

Note: The accumulated Hebbian weights (_hebbian_weights) are not saved in checkpoints
due to a bug (not registered as buffer). This script visualizes:
- base_lateral: Learnable lateral weights trained via backprop
- lateral_gate_weights: Per-neuron gating of lateral input
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def extract_step(filename):
    """Extract step number from checkpoint filename."""
    match = re.search(r'checkpoint_(\d+)\.pt', filename)
    if match:
        return int(match.group(1))
    return 0


def load_lateral_weights(checkpoint_path):
    """Load lateral connection weights from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    base_lateral = None
    lateral_gate = None
    hebbian_weights = None

    for key, value in state_dict.items():
        if 'hebbian.base_lateral' in key:
            base_lateral = value.numpy()
        elif 'hebbian.lateral_gate_weights' in key:
            lateral_gate = value.numpy()
        elif 'hebbian._hebbian_weights' in key and value is not None:
            hebbian_weights = value.numpy()

    return base_lateral, lateral_gate, hebbian_weights


def create_frame(base_lateral, lateral_gate, step, output_path, title_prefix=""):
    """Create a single frame visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Base lateral weights (learned via backprop)
    if base_lateral is not None:
        # Zero diagonal for display
        display_weights = base_lateral.copy()
        np.fill_diagonal(display_weights, 0)

        vmax = np.abs(display_weights).max() or 1
        im0 = axes[0].imshow(display_weights, cmap='RdBu_r', aspect='auto',
                             vmin=-vmax, vmax=vmax)
        axes[0].set_title(f'{title_prefix}Base Lateral Weights\nStep {step} (max={vmax:.4f})')
        axes[0].set_xlabel('Neuron j')
        axes[0].set_ylabel('Neuron i')
        plt.colorbar(im0, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, 'Not available',
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f'{title_prefix}Base Lateral Weights\nStep {step}')

    # Lateral gate weights (sigmoid applied for interpretation)
    if lateral_gate is not None:
        gate_values = 1 / (1 + np.exp(-lateral_gate))  # sigmoid

        # Sort neurons by gate value for better visualization
        sorted_idx = np.argsort(gate_values)

        axes[1].bar(range(len(gate_values)), gate_values[sorted_idx],
                   color=plt.cm.RdYlGn(gate_values[sorted_idx]), width=1.0)
        axes[1].set_title(f'Lateral Gate (sigmoid)\nmean={gate_values.mean():.3f}, std={gate_values.std():.3f}')
        axes[1].set_xlabel('Neuron (sorted by gate value)')
        axes[1].set_ylabel('Gate Value')
        axes[1].set_ylim([0, 1])
        axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, 'Not available',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Lateral Gate')

    # Weight distribution
    if base_lateral is not None:
        flat_weights = base_lateral.flatten()
        axes[2].hist(flat_weights, bins=100, alpha=0.7, color='blue', density=True)
        axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title(f'Weight Distribution\nmean={flat_weights.mean():.4f}, std={flat_weights.std():.4f}')
        axes[2].set_xlabel('Weight Value')
        axes[2].set_ylabel('Density')

        # Add stats
        nonzero_frac = (np.abs(flat_weights) > 1e-6).mean()
        stats_text = f'max: {flat_weights.max():.4f}\nmin: {flat_weights.min():.4f}\nnon-zero: {nonzero_frac:.1%}'
        axes[2].text(0.95, 0.95, stats_text, transform=axes[2].transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[2].text(0.5, 0.5, 'Not available',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Weight Distribution')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def create_gif(frame_paths, output_path, duration=200):
    """Create a GIF from frame images."""
    frames = [Image.open(fp) for fp in frame_paths]
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Created GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize lateral weight evolution')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for frames and GIF')
    parser.add_argument('--title_prefix', type=str, default='',
                        help='Prefix for plot titles')
    parser.add_argument('--duration', type=int, default=200,
                        help='GIF frame duration in ms')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process every N-th checkpoint')
    args = parser.parse_args()

    # Find checkpoints
    checkpoint_pattern = os.path.join(args.checkpoint_dir, 'checkpoint_*.pt')
    checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=extract_step)

    if not checkpoint_files:
        print(f"No checkpoints found matching: {checkpoint_pattern}")
        return

    print(f"Found {len(checkpoint_files)} checkpoints")

    if args.skip > 1:
        checkpoint_files = checkpoint_files[::args.skip]
        print(f"Processing every {args.skip}th checkpoint: {len(checkpoint_files)} total")

    # Setup output
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, 'lateral_evolution')
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Track stats over time
    steps = []
    max_weights = []
    std_weights = []
    mean_gates = []

    # Process checkpoints
    frame_paths = []
    for checkpoint_path in tqdm(checkpoint_files, desc='Processing checkpoints'):
        step = extract_step(checkpoint_path)
        frame_path = os.path.join(frames_dir, f'frame_{step:06d}.png')

        try:
            base_lateral, lateral_gate, _ = load_lateral_weights(checkpoint_path)
            create_frame(base_lateral, lateral_gate, step, frame_path, args.title_prefix)
            frame_paths.append(frame_path)

            # Collect stats
            if base_lateral is not None:
                steps.append(step)
                max_weights.append(np.abs(base_lateral).max())
                std_weights.append(base_lateral.std())
            if lateral_gate is not None:
                gate_values = 1 / (1 + np.exp(-lateral_gate))
                mean_gates.append(gate_values.mean())

        except Exception as e:
            print(f"Error processing {checkpoint_path}: {e}")
            continue

    # Create GIF
    if frame_paths:
        gif_path = os.path.join(output_dir, 'lateral_evolution.gif')
        create_gif(frame_paths, gif_path, args.duration)

    # Create summary plot
    if steps:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Weight stats over time
        axes[0, 0].plot(steps, max_weights, 'r-', label='Max |weight|')
        axes[0, 0].plot(steps, std_weights, 'b-', label='Std')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Weight Value')
        axes[0, 0].set_title('Base Lateral Weight Statistics')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Gate mean over time
        if mean_gates:
            axes[0, 1].plot(steps[:len(mean_gates)], mean_gates, 'g-')
            axes[0, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Mean Gate Value')
            axes[0, 1].set_title('Lateral Gate Evolution')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3)

        # Early vs Late comparison
        early_weights, early_gate, _ = load_lateral_weights(checkpoint_files[0])
        late_weights, late_gate, _ = load_lateral_weights(checkpoint_files[-1])

        if early_weights is not None and late_weights is not None:
            early_step = extract_step(checkpoint_files[0])
            late_step = extract_step(checkpoint_files[-1])

            # Show difference
            diff = late_weights - early_weights
            np.fill_diagonal(diff, 0)
            vmax = np.abs(diff).max() or 1

            im = axes[1, 0].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            axes[1, 0].set_title(f'Change in Weights\n(Step {late_step} - Step {early_step})')
            axes[1, 0].set_xlabel('Neuron j')
            axes[1, 0].set_ylabel('Neuron i')
            plt.colorbar(im, ax=axes[1, 0])

            # Late weights
            np.fill_diagonal(late_weights, 0)
            vmax_late = np.abs(late_weights).max() or 1
            im2 = axes[1, 1].imshow(late_weights, cmap='RdBu_r', aspect='auto',
                                    vmin=-vmax_late, vmax=vmax_late)
            axes[1, 1].set_title(f'Final Weights (Step {late_step})\nmax={vmax_late:.4f}')
            axes[1, 1].set_xlabel('Neuron j')
            axes[1, 1].set_ylabel('Neuron i')
            plt.colorbar(im2, ax=axes[1, 1])

        plt.tight_layout()
        summary_path = os.path.join(output_dir, 'lateral_summary.png')
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"Created summary: {summary_path}")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
