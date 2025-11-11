"""
Standalone script to generate all plots from existing training/evaluation results.
This can be run independently without needing the environment.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import plot_training_curves, plot_evaluation_results, plot_metrics, load_results


def generate_all_plots(results_dir='results', logs_dir='logs'):
    """Generate all required plots from existing results."""
    layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating all required plots")
    print("="*60)
    
    # 1. MANDATORY: Training curves for all layouts
    print("\n1. Generating training curves (mandatory)...")
    training_data = {}
    for layout in layouts:
        log_path = Path(logs_dir) / f"{layout}_training_results.json"
        if log_path.exists():
            try:
                results = load_results(log_path)
                if 'soups_delivered' in results and len(results['soups_delivered']) > 0:
                    training_data[layout] = results['soups_delivered']
                    print(f"   Loaded {len(results['soups_delivered'])} episodes for {layout}")
            except Exception as e:
                print(f"   [ERROR] Failed to load {log_path}: {e}")
    
    if training_data:
        plot_path = Path(results_dir) / "training_curves_all_layouts.png"
        try:
            plot_training_curves(
                training_data,
                save_path=plot_path,
                title="Training Progress: Soups Delivered Across All Layouts"
            )
            if plot_path.exists():
                print(f"   [OK] Saved: {plot_path}")
            else:
                print(f"   [FAIL] File not created: {plot_path}")
        except Exception as e:
            print(f"   [ERROR] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   [WARNING] No training data found")
    
    # 2. MANDATORY: Evaluation curves for all layouts
    print("\n2. Generating evaluation curves (mandatory)...")
    eval_data = {}
    for layout in layouts:
        result_path = Path(results_dir) / f"{layout}_evaluation.json"
        if result_path.exists():
            try:
                results = load_results(result_path)
                if 'episode_soups' in results and len(results['episode_soups']) > 0:
                    eval_data[layout] = results['episode_soups']
                    print(f"   Loaded {len(results['episode_soups'])} episodes for {layout}")
            except Exception as e:
                print(f"   [ERROR] Failed to load {result_path}: {e}")
    
    if eval_data:
        plot_path = Path(results_dir) / "evaluation_curves_all_layouts.png"
        try:
            plot_evaluation_results(
                eval_data,
                save_path=plot_path,
                title="Evaluation Results: Soups Delivered Per Episode (≥100 Episodes)"
            )
            if plot_path.exists():
                print(f"   [OK] Saved: {plot_path}")
            else:
                print(f"   [FAIL] File not created: {plot_path}")
        except Exception as e:
            print(f"   [ERROR] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   [WARNING] No evaluation data found")
    
    # 3. MANDATORY: At least 2 auxiliary metrics plots
    print("\n3. Generating auxiliary metrics plots (mandatory: ≥2)...")
    
    # Plot 1: Collaboration metrics
    print("   a. Collaboration metrics (onion & dish pickups)...")
    collaboration_metrics = {}
    for layout in layouts:
        log_path = Path(logs_dir) / f"{layout}_training_results.json"
        if log_path.exists():
            try:
                results = load_results(log_path)
                if 'auxiliary_metrics' in results:
                    aux = results['auxiliary_metrics']
                    if 'onion_pickups' in aux and len(aux['onion_pickups']) > 0:
                        collaboration_metrics[f"{layout}_onion"] = aux['onion_pickups']
                    if 'dish_pickups' in aux and len(aux['dish_pickups']) > 0:
                        collaboration_metrics[f"{layout}_dish"] = aux['dish_pickups']
            except Exception as e:
                print(f"      [ERROR] Failed to load metrics for {layout}: {e}")
    
    if collaboration_metrics:
        plot_path = Path(results_dir) / "auxiliary_collaboration_metrics.png"
        try:
            plot_metrics(
                collaboration_metrics,
                save_path=plot_path,
                title="Auxiliary Metrics: Onion & Dish Pickups (Collaboration)"
            )
            if plot_path.exists():
                print(f"      [OK] Saved: {plot_path}")
            else:
                print(f"      [FAIL] File not created: {plot_path}")
        except Exception as e:
            print(f"      [ERROR] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("      [WARNING] No collaboration metrics found")
    
    # Plot 2: Efficiency metrics
    print("   b. Efficiency metrics (placements & soup pickups)...")
    efficiency_metrics = {}
    for layout in layouts:
        log_path = Path(logs_dir) / f"{layout}_training_results.json"
        if log_path.exists():
            try:
                results = load_results(log_path)
                if 'auxiliary_metrics' in results:
                    aux = results['auxiliary_metrics']
                    if 'placements_in_pot' in aux and len(aux['placements_in_pot']) > 0:
                        efficiency_metrics[f"{layout}_placements"] = aux['placements_in_pot']
                    if 'soup_pickups' in aux and len(aux['soup_pickups']) > 0:
                        efficiency_metrics[f"{layout}_soup_pickups"] = aux['soup_pickups']
            except Exception as e:
                print(f"      [ERROR] Failed to load metrics for {layout}: {e}")
    
    if efficiency_metrics:
        plot_path = Path(results_dir) / "auxiliary_efficiency_metrics.png"
        try:
            plot_metrics(
                efficiency_metrics,
                save_path=plot_path,
                title="Auxiliary Metrics: Placements & Soup Pickups (Efficiency)"
            )
            if plot_path.exists():
                print(f"      [OK] Saved: {plot_path}")
            else:
                print(f"      [FAIL] File not created: {plot_path}")
        except Exception as e:
            print(f"      [ERROR] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("      [WARNING] No efficiency metrics found")
    
    # Summary
    print("\n" + "="*60)
    print("Plot generation completed!")
    print("="*60)
    
    # List all generated files
    plot_files = list(Path(results_dir).glob("*.png"))
    if plot_files:
        print(f"\nGenerated {len(plot_files)} plot files:")
        for f in sorted(plot_files):
            size = f.stat().st_size
            print(f"  - {f.name} ({size:,} bytes)")
    else:
        print("\nNo plot files found in results directory")


def main():
    parser = argparse.ArgumentParser(description='Generate all plots from existing results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing evaluation results')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory containing training logs')
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.logs_dir)


if __name__ == '__main__':
    main()

