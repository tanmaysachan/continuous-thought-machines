"""
Compare Maze Solving Results Across Models

Loads results from baseline, hebbian, and hebbian-oja analysis runs
and creates comparative visualizations focusing on reapplication counts.

Usage:
    python scripts/compare_maze_results.py \
        --results_dir logs/baseline_hebbian_hebbian_oja_comparative_analysis
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')


def load_results(results_path):
    """Load results from npz file."""
    data = np.load(results_path, allow_pickle=True)
    return data['results'].item()


def main():
    parser = argparse.ArgumentParser(description='Compare maze solving results')
    parser.add_argument('--results_dir', type=str,
                        default='logs/baseline_hebbian_hebbian_oja_comparative_analysis',
                        help='Directory containing model results')
    parser.add_argument('--dataset', type=str, default='medium',
                        help='Dataset name used in analysis')
    args = parser.parse_args()

    # Load results from each model
    models = {
        'Baseline': os.path.join(args.results_dir, 'baseline', 'gen', args.dataset, f'{args.dataset}_results.npz'),
        'Hebbian': os.path.join(args.results_dir, 'hebbian', 'gen', args.dataset, f'{args.dataset}_results.npz'),
        'Hebbian-Oja': os.path.join(args.results_dir, 'hebbian_oja', 'gen', args.dataset, f'{args.dataset}_results.npz'),
    }

    results = {}
    for name, path in models.items():
        if os.path.exists(path):
            results[name] = load_results(path)
            print(f"Loaded {name}: {len(results[name])} route lengths")
        else:
            print(f"Warning: {path} not found")

    if not results:
        print("No results found!")
        return

    # Create output directory
    output_dir = os.path.join(args.results_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    # Get all route lengths across all models
    all_lengths = set()
    for model_results in results.values():
        all_lengths.update(model_results.keys())
    sorted_lengths = sorted(all_lengths)

    # Prepare data for plotting
    palette = sns.color_palette("husl", len(results))

    # === Figure 1: Success Rate Comparison ===
    fig_success, ax_success = plt.subplots(figsize=(10, 6))

    for i, (name, model_results) in enumerate(results.items()):
        lengths = sorted(model_results.keys())
        success_rates = [np.mean([r[0] for r in model_results[l]]) * 100 for l in lengths]
        ax_success.plot(lengths, success_rates, '-o', label=name, color=palette[i], markersize=4)

    ax_success.set_xlabel('Route Length', fontsize=12)
    ax_success.set_ylabel('Success Rate (%)', fontsize=12)
    ax_success.set_title('Maze Solving Success Rate by Route Length', fontsize=14)
    ax_success.legend()
    ax_success.set_ylim([0, 105])
    ax_success.grid(True, alpha=0.3)

    fig_success.tight_layout()
    fig_success.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=200)
    fig_success.savefig(os.path.join(output_dir, 'success_rate_comparison.pdf'))
    plt.close(fig_success)

    # === Figure 2: Reapplications Comparison (Main Interest) ===
    fig_reapp, ax_reapp = plt.subplots(figsize=(10, 6))

    for i, (name, model_results) in enumerate(results.items()):
        lengths = sorted(model_results.keys())
        # Only count reapplications for solved mazes
        reapps_mean = []
        for l in lengths:
            solved_reapps = [r[1] for r in model_results[l] if r[0]]
            reapps_mean.append(np.mean(solved_reapps) if solved_reapps else np.nan)
        ax_reapp.plot(lengths, reapps_mean, '-o', label=name, color=palette[i], markersize=4)

    ax_reapp.set_xlabel('Route Length', fontsize=12)
    ax_reapp.set_ylabel('Average Reapplications (on solved)', fontsize=12)
    ax_reapp.set_title('Reapplications Needed to Solve Maze', fontsize=14)
    ax_reapp.legend()
    ax_reapp.grid(True, alpha=0.3)

    fig_reapp.tight_layout()
    fig_reapp.savefig(os.path.join(output_dir, 'reapplications_comparison.png'), dpi=200)
    fig_reapp.savefig(os.path.join(output_dir, 'reapplications_comparison.pdf'))
    plt.close(fig_reapp)

    # === Figure 3: Efficiency (Success per Reapplication) ===
    fig_eff, ax_eff = plt.subplots(figsize=(10, 6))

    for i, (name, model_results) in enumerate(results.items()):
        lengths = sorted(model_results.keys())
        efficiency = []
        for l in lengths:
            solved = [r for r in model_results[l] if r[0]]
            if solved:
                # Efficiency = 1 / avg_reapplications (higher is better)
                avg_reapp = np.mean([r[1] for r in solved])
                efficiency.append(1.0 / avg_reapp if avg_reapp > 0 else 0)
            else:
                efficiency.append(0)
        ax_eff.plot(lengths, efficiency, '-o', label=name, color=palette[i], markersize=4)

    ax_eff.set_xlabel('Route Length', fontsize=12)
    ax_eff.set_ylabel('Efficiency (1 / Reapplications)', fontsize=12)
    ax_eff.set_title('Solving Efficiency (Higher = Fewer Reapplications)', fontsize=14)
    ax_eff.legend()
    ax_eff.grid(True, alpha=0.3)

    fig_eff.tight_layout()
    fig_eff.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=200)
    fig_eff.savefig(os.path.join(output_dir, 'efficiency_comparison.pdf'))
    plt.close(fig_eff)

    # === Summary Statistics ===
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*60)

    for name, model_results in results.items():
        all_results = []
        for l, res_list in model_results.items():
            all_results.extend(res_list)

        total = len(all_results)
        solved = sum(1 for r in all_results if r[0])
        solved_reapps = [r[1] for r in all_results if r[0]]

        print(f"\n{name}:")
        print(f"  Total mazes tested: {total}")
        print(f"  Solved: {solved} ({100*solved/total:.1f}%)")
        if solved_reapps:
            print(f"  Avg reapplications (solved): {np.mean(solved_reapps):.2f}")
            print(f"  Median reapplications (solved): {np.median(solved_reapps):.1f}")
            print(f"  Min/Max reapplications: {min(solved_reapps)}/{max(solved_reapps)}")

    # === Save summary to file ===
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("COMPARATIVE ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")

        for name, model_results in results.items():
            all_results = []
            for l, res_list in model_results.items():
                all_results.extend(res_list)

            total = len(all_results)
            solved = sum(1 for r in all_results if r[0])
            solved_reapps = [r[1] for r in all_results if r[0]]

            f.write(f"{name}:\n")
            f.write(f"  Total mazes tested: {total}\n")
            f.write(f"  Solved: {solved} ({100*solved/total:.1f}%)\n")
            if solved_reapps:
                f.write(f"  Avg reapplications (solved): {np.mean(solved_reapps):.2f}\n")
                f.write(f"  Median reapplications (solved): {np.median(solved_reapps):.1f}\n")
                f.write(f"  Min/Max reapplications: {min(solved_reapps)}/{max(solved_reapps)}\n")
            f.write("\n")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - success_rate_comparison.png")
    print(f"  - reapplications_comparison.png")
    print(f"  - efficiency_comparison.png")
    print(f"  - summary.txt")


if __name__ == '__main__':
    main()
