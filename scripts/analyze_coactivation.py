"""
Analyze Neuron Co-activation Patterns

Runs inference on test data and computes pairwise neuron correlations
to understand which neurons fire together - the signal that drives Hebbian learning.

This reconstructs what _hebbian_weights would capture, even though they aren't saved.

Usage:
    python scripts/analyze_coactivation.py \
        --checkpoint logs/hebbian/mazes-small-mac/checkpoint_46000.pt \
        --n_samples 100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data.custom_datasets import MazeImageFolder
from models.ctm import ContinuousThoughtMachine
from models.ctm_hebbian import wrap_ctm_with_hebbian


def load_model(checkpoint_path, device='cpu'):
    """Load HebbianCTM from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt['args']

    # Recreate model architecture
    prediction_reshaper = [args.maze_route_length, 5]
    out_dims = args.maze_route_length * 5

    base_ctm = ContinuousThoughtMachine(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=args.backbone_type,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
    )

    model = wrap_ctm_with_hebbian(
        base_ctm,
        hebbian_lr=args.hebbian_lr,
        hebbian_decay=args.hebbian_decay,
        use_oja=args.use_oja,
        lateral_strength=args.lateral_strength,
        lateral_injection=args.lateral_injection,
    ).to(device)

    # Initialize lazy modules
    h_w = 39 if 'small' in args.dataset or 'medium' in args.dataset else 99
    pseudo_inputs = torch.zeros((1, 3, h_w, h_w), device=device).float()
    model(pseudo_inputs)

    # Load weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, args


def collect_activations(model, dataloader, n_samples, device):
    """Run inference and collect neuron activations."""
    all_activations = []  # List of (n_iterations, d_model) per sample

    n_collected = 0
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc='Collecting activations'):
            if n_collected >= n_samples:
                break

            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Run with tracking
            results = model(inputs, track=True)
            # results[4] is post_activations: (n_iterations, batch, d_model)
            post_activations = results[4]  # numpy array

            # Transpose to (batch, n_iterations, d_model)
            post_activations = np.transpose(post_activations, (1, 0, 2))

            for i in range(batch_size):
                if n_collected >= n_samples:
                    break
                all_activations.append(post_activations[i])
                n_collected += 1

    return np.array(all_activations)  # (n_samples, n_iterations, d_model)


def compute_coactivation_matrix(activations):
    """
    Compute neuron co-activation patterns.

    Args:
        activations: (n_samples, n_iterations, d_model)

    Returns:
        correlation_matrix: (d_model, d_model) pairwise correlations
        coactivation_matrix: (d_model, d_model) mean pairwise products (Hebbian signal)
    """
    n_samples, n_iterations, d_model = activations.shape

    # Flatten samples and iterations: (n_samples * n_iterations, d_model)
    flat_activations = activations.reshape(-1, d_model)

    # Correlation matrix
    # Normalize each neuron's activations
    means = flat_activations.mean(axis=0, keepdims=True)
    stds = flat_activations.std(axis=0, keepdims=True) + 1e-8
    normalized = (flat_activations - means) / stds
    correlation_matrix = (normalized.T @ normalized) / flat_activations.shape[0]

    # Co-activation matrix (what Hebbian learning sees)
    # Mean of outer products: E[act_i * act_j]
    coactivation_matrix = (flat_activations.T @ flat_activations) / flat_activations.shape[0]

    return correlation_matrix, coactivation_matrix


def analyze_temporal_coactivation(activations):
    """
    Analyze how co-activation evolves over iterations within a sample.

    Args:
        activations: (n_samples, n_iterations, d_model)

    Returns:
        early_coact: Co-activation in first 1/3 of iterations
        late_coact: Co-activation in last 1/3 of iterations
    """
    n_iterations = activations.shape[1]
    third = n_iterations // 3

    early = activations[:, :third, :]
    late = activations[:, -third:, :]

    _, early_coact = compute_coactivation_matrix(early)
    _, late_coact = compute_coactivation_matrix(late)

    return early_coact, late_coact


def main():
    parser = argparse.ArgumentParser(description='Analyze neuron co-activation patterns')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization')
    parser.add_argument('--data_root', type=str, default='data/mazes',
                        help='Data root directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda:0)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, train_args = load_model(args.checkpoint, args.device)

    # Load test data
    which_maze = train_args.dataset.split('-')[-1]
    data_root = f'{args.data_root}/{which_maze}'

    test_data = MazeImageFolder(
        root=f'{data_root}/test/',
        which_set='test',
        maze_route_length=train_args.maze_route_length,
        expand_range=train_args.expand_range
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=16, shuffle=True, num_workers=0
    )

    # Collect activations
    print(f"Collecting activations from {args.n_samples} samples...")
    activations = collect_activations(model, testloader, args.n_samples, args.device)
    print(f"Activations shape: {activations.shape}")

    # Compute co-activation matrices
    print("Computing co-activation patterns...")
    correlation, coactivation = compute_coactivation_matrix(activations)
    early_coact, late_coact = analyze_temporal_coactivation(activations)

    # Zero diagonals for visualization
    np.fill_diagonal(correlation, 0)
    np.fill_diagonal(coactivation, 0)
    np.fill_diagonal(early_coact, 0)
    np.fill_diagonal(late_coact, 0)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    step = train_args.training_iterations if hasattr(train_args, 'training_iterations') else 'unknown'
    fig.suptitle(f'Neuron Co-activation Analysis (Step {step}, {args.n_samples} samples)', fontsize=14)

    # Row 1: Full matrices
    # 1a. Correlation matrix (zoomed)
    zoom = 100
    corr_zoom = correlation[:zoom, :zoom]
    vmax = np.abs(corr_zoom).max() or 1
    im0 = axes[0, 0].imshow(corr_zoom, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title(f'Correlation Matrix (neurons 0-{zoom})')
    axes[0, 0].set_xlabel('Neuron j')
    axes[0, 0].set_ylabel('Neuron i')
    plt.colorbar(im0, ax=axes[0, 0])

    # 1b. Co-activation matrix (Hebbian signal) - zoomed
    coact_zoom = coactivation[:zoom, :zoom]
    vmax_coact = np.abs(coact_zoom).max() or 1
    im1 = axes[0, 1].imshow(coact_zoom, cmap='RdBu_r', vmin=-vmax_coact, vmax=vmax_coact)
    axes[0, 1].set_title(f'Co-activation (Hebbian signal)\nneurons 0-{zoom}')
    axes[0, 1].set_xlabel('Neuron j')
    axes[0, 1].set_ylabel('Neuron i')
    plt.colorbar(im1, ax=axes[0, 1])

    # 1c. Distribution of co-activation values
    flat_coact = coactivation.flatten()
    axes[0, 2].hist(flat_coact[flat_coact != 0], bins=100, alpha=0.7, color='purple')
    axes[0, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].set_title(f'Co-activation Distribution\nmean={flat_coact.mean():.4f}, std={flat_coact.std():.4f}')
    axes[0, 2].set_xlabel('Co-activation Value')
    axes[0, 2].set_yscale('log')

    # Row 2: Temporal and comparison
    # 2a. Early iterations co-activation
    early_zoom = early_coact[:zoom, :zoom]
    vmax_e = np.abs(early_zoom).max() or 1
    im2 = axes[1, 0].imshow(early_zoom, cmap='RdBu_r', vmin=-vmax_e, vmax=vmax_e)
    axes[1, 0].set_title(f'Early Iterations (first 1/3)\nmax={vmax_e:.4f}')
    axes[1, 0].set_xlabel('Neuron j')
    axes[1, 0].set_ylabel('Neuron i')
    plt.colorbar(im2, ax=axes[1, 0])

    # 2b. Late iterations co-activation
    late_zoom = late_coact[:zoom, :zoom]
    vmax_l = np.abs(late_zoom).max() or 1
    im3 = axes[1, 1].imshow(late_zoom, cmap='RdBu_r', vmin=-vmax_l, vmax=vmax_l)
    axes[1, 1].set_title(f'Late Iterations (last 1/3)\nmax={vmax_l:.4f}')
    axes[1, 1].set_xlabel('Neuron j')
    axes[1, 1].set_ylabel('Neuron i')
    plt.colorbar(im3, ax=axes[1, 1])

    # 2c. Difference (late - early)
    diff = late_coact - early_coact
    diff_zoom = diff[:zoom, :zoom]
    vmax_d = np.abs(diff_zoom).max() or 1
    im4 = axes[1, 2].imshow(diff_zoom, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    axes[1, 2].set_title(f'Change (Late - Early)\nmax_change={vmax_d:.4f}')
    axes[1, 2].set_xlabel('Neuron j')
    axes[1, 2].set_ylabel('Neuron i')
    plt.colorbar(im4, ax=axes[1, 2])

    plt.tight_layout()

    # Save
    if args.output:
        output_path = args.output
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
        output_path = os.path.join(ckpt_dir, f'{ckpt_name}_coactivation.png')

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # Print statistics
    print(f"\n=== Co-activation Summary ===")
    print(f"Correlation range: [{correlation.min():.4f}, {correlation.max():.4f}]")
    print(f"Co-activation range: [{coactivation.min():.4f}, {coactivation.max():.4f}]")
    print(f"Strong correlations (|r| > 0.3): {(np.abs(correlation) > 0.3).sum()}")
    print(f"Strong co-activations (|c| > 0.1): {(np.abs(coactivation) > 0.1).sum()}")

    # Top co-activating pairs
    print(f"\nTop 10 co-activating neuron pairs:")
    flat_idx = np.argsort(coactivation.flatten())[::-1]
    for i in range(10):
        idx = flat_idx[i]
        ni, nj = idx // coactivation.shape[1], idx % coactivation.shape[1]
        if ni != nj:
            print(f"  Neurons ({ni}, {nj}): {coactivation[ni, nj]:.4f}")


if __name__ == '__main__':
    main()
