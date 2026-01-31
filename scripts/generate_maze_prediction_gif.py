#!/usr/bin/env python
"""
Generate prediction.gif from a maze CTM checkpoint.

This script loads a checkpoint and generates the prediction.gif visualization
showing the model's attention trajectory through the maze as arrows.

Usage:
    python scripts/generate_maze_prediction_gif.py --checkpoint logs/hebbian/mazes-small-mac/checkpoint.pt
    python scripts/generate_maze_prediction_gif.py --checkpoint logs/hebbian/mazes-small-mac/checkpoint_5000.pt --output_dir outputs/viz
"""

# Set matplotlib backend before any other imports to avoid Retina display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Fix for Retina display DPI scaling issues
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

import torch
import numpy as np
import os
import argparse

from data.custom_datasets import MazeImageFolder
from models.ctm import ContinuousThoughtMachine
from models.ctm_hebbian import wrap_ctm_with_hebbian
from tasks.mazes.plotting import make_maze_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prediction.gif from maze CTM checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument('--device', type=str, default=None, help="Device: 'cuda', 'mps', 'cpu' (auto-detect if not specified)")
    parser.add_argument('--sample_index', type=int, default=None, help="Index of sample to visualize (default: longest path)")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset override (default: from checkpoint args)")
    parser.add_argument('--data_root', type=str, default='data/mazes', help="Root directory for maze data")
    parser.add_argument('--legacy_scaling', action=argparse.BooleanOptionalAction, default=False,
                        help='Use [0,1] scaling instead of [-1,1] (for old checkpoints)')
    return parser.parse_args()


def get_device(device_arg):
    """Auto-detect best available device."""
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model_from_checkpoint(checkpoint_path, device):
    """Load CTM or HebbianCTM from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']

    # Handle legacy arguments
    if not hasattr(model_args, 'backbone_type') and hasattr(model_args, 'resnet_type'):
        model_args.backbone_type = f'{model_args.resnet_type}-{getattr(model_args, "resnet_feature_scales", [4])[-1]}'
    if not hasattr(model_args, 'neuron_select_type'):
        model_args.neuron_select_type = 'first-last'
    if not hasattr(model_args, 'n_random_pairing_self'):
        model_args.n_random_pairing_self = 0
    if not hasattr(model_args, 'dropout_nlm'):
        model_args.dropout_nlm = 0

    # Build prediction reshaper
    prediction_reshaper = [model_args.out_dims // 5, 5] if hasattr(model_args, 'out_dims') else None

    print("Instantiating CTM model...")
    base_ctm = ContinuousThoughtMachine(
        iterations=model_args.iterations,
        d_model=model_args.d_model,
        d_input=model_args.d_input,
        heads=model_args.heads,
        n_synch_out=model_args.n_synch_out,
        n_synch_action=model_args.n_synch_action,
        synapse_depth=model_args.synapse_depth,
        memory_length=model_args.memory_length,
        deep_nlms=getattr(model_args, 'deep_memory', False),
        memory_hidden_dims=getattr(model_args, 'memory_hidden_dims', 64),
        do_layernorm_nlm=getattr(model_args, 'do_normalisation', False),
        backbone_type=model_args.backbone_type,
        positional_embedding_type=getattr(model_args, 'positional_embedding_type', 'learnable'),
        out_dims=model_args.out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=0,  # No dropout for inference
        dropout_nlm=0,
        neuron_select_type=model_args.neuron_select_type,
        n_random_pairing_self=model_args.n_random_pairing_self,
    )

    # Check if this is a Hebbian checkpoint
    use_hebbian = getattr(model_args, 'use_hebbian', False)
    if use_hebbian:
        print(f"Wrapping with Hebbian (lr={model_args.hebbian_lr}, decay={model_args.hebbian_decay})")
        model = wrap_ctm_with_hebbian(
            base_ctm,
            hebbian_lr=getattr(model_args, 'hebbian_lr', 0.0001),
            hebbian_decay=getattr(model_args, 'hebbian_decay', 0.999),
            use_oja=getattr(model_args, 'use_oja', False),
            lateral_strength=getattr(model_args, 'lateral_strength', 0.1),
            lateral_injection=getattr(model_args, 'lateral_injection', 'pre_synapse'),
        ).to(device)
    else:
        model = base_ctm.to(device)

    # Load weights
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
    load_result = model.load_state_dict(checkpoint[state_dict_key], strict=False)
    print(f"Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

    model.eval()
    return model, model_args


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine dataset
    if args.dataset:
        dataset = args.dataset
    else:
        dataset = getattr(model_args, 'dataset', 'mazes-small')

    which_maze = dataset.split('-')[-1]
    data_root = f'{args.data_root}/{which_maze}'
    print(f"Loading test data from: {data_root}")

    # Load test data
    maze_route_length = getattr(model_args, 'maze_route_length', 50)
    expand_range = not args.legacy_scaling

    test_data = MazeImageFolder(
        root=f'{data_root}/test/',
        which_set='test',
        maze_route_length=maze_route_length,
        expand_range=expand_range
    )

    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )

    # Get a batch
    inputs, targets = next(iter(testloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Select sample to visualize
    if args.sample_index is not None:
        sample_idx = args.sample_index
    else:
        # Find longest path in batch
        sample_idx = (targets != 4).sum(-1).argmax().item()

    print(f"Visualizing sample {sample_idx} (path length: {(targets[sample_idx] != 4).sum().item()})")

    # Run inference with tracking
    print("Running inference with tracking...")
    with torch.no_grad():
        predictions_raw, certainties, _, pre_activations, post_activations, attention_tracking = model(inputs, track=True)

    # Reshape predictions: (B, D, T) -> (B, S, C, T)
    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))

    # Reshape attention tracking
    base_model = model.ctm if hasattr(model, 'ctm') else model
    att_shape = (base_model.kv_features.shape[2], base_model.kv_features.shape[3])
    attention_tracking = attention_tracking.reshape(
        attention_tracking.shape[0],  # iterations
        attention_tracking.shape[1],  # batch
        -1,  # heads
        att_shape[0],
        att_shape[1]
    )

    # Prepare data for make_maze_gif
    # Rescale inputs from [-1, 1] to [0, 1] if using new scaling
    if expand_range:
        inputs_viz = (inputs[sample_idx].detach().cpu().numpy() + 1) / 2
    else:
        inputs_viz = inputs[sample_idx].detach().cpu().numpy()

    predictions_viz = predictions[sample_idx].detach().cpu().numpy()  # (S, C, T)
    targets_viz = targets[sample_idx].detach().cpu().numpy()  # (S,)
    attention_viz = attention_tracking[:, sample_idx]  # (T, H, h, w)

    print(f"Generating prediction.gif...")
    print(f"  Input shape: {inputs_viz.shape}")
    print(f"  Predictions shape: {predictions_viz.shape}")
    print(f"  Targets shape: {targets_viz.shape}")
    print(f"  Attention shape: {attention_viz.shape}")

    make_maze_gif(
        inputs_viz,
        predictions_viz,
        targets_viz,
        attention_viz,
        output_dir,
        verbose=True
    )

    print(f"\nDone! Output saved to:")
    print(f"  {output_dir}/prediction.gif")
    print(f"  {output_dir}/route_approximation.png")


if __name__ == '__main__':
    main()
