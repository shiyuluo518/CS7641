"""
Convenience script to train on all layouts sequentially.

QUICK START:
============
This script trains all three layouts in QUICK MODE by default (~20-30 minutes, ULTRA-FAST).
To switch to FULL MODE (≥7.0 performance guarantee, ~10-13 hours), edit src/config.py:
    QUICK_MODE = False

QUICK MODE (Default - ULTRA-FAST):
- cramped_room: 1,000 episodes (~5-8 min)
- coordination_ring: 1,500 episodes (~10-15 min)
- counter_circuit_o_1order: 1,500 episodes (~10-15 min)
Total: ~20-30 minutes

FULL MODE (QUICK_MODE = False):
- cramped_room: 15,000 episodes (~2-3 hours)
- coordination_ring: 25,000 episodes (~4-5 hours)
- counter_circuit_o_1order: 25,000 episodes (~4-5 hours)
Total: ~10-13 hours

After training, run:
    1. python src/evaluate.py --all
    2. python verify_results.py
"""

import subprocess
import sys
from pathlib import Path

layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']

def train_all():
    """Train on all three layouts sequentially."""
    # Check if quick mode is enabled
    import src.config as config
    mode = "QUICK" if config.QUICK_MODE else "FULL"
    
    print("="*60)
    print(f"Training on all layouts ({mode} MODE)")
    print("="*60)
    if config.QUICK_MODE:
        print("[QUICK MODE] Ultra-fast mode (~20-30 minutes total)")
        print("  Set QUICK_MODE = False in src/config.py for full training (~10-13 hours)")
        print(f"  - cramped_room: {config.EPISODES_CRAMPED_ROOM} episodes (~5-8 min)")
        print(f"  - coordination_ring: {config.EPISODES_COORDINATION_RING} episodes (~10-15 min)")
        print(f"  - counter_circuit_o_1order: {config.EPISODES_COUNTER_CIRCUIT} episodes (~10-15 min)")
        print("  Ultra-fast optimizations: small batch/buffer, train every step, minimal logging")
    else:
        print("[FULL MODE] Full episodes for >=7.0 performance guarantee (~10-13 hours)")
        print(f"  - cramped_room: {config.EPISODES_CRAMPED_ROOM} episodes (~2-3 hours)")
        print(f"  - coordination_ring: {config.EPISODES_COORDINATION_RING} episodes (~4-5 hours)")
        print(f"  - counter_circuit_o_1order: {config.EPISODES_COUNTER_CIRCUIT} episodes (~4-5 hours)")
    print("="*60)
    
    for layout in layouts:
        print(f"\n{'='*60}")
        print(f"Training on {layout}")
        print(f"{'='*60}\n")
        
        cmd = [sys.executable, 'src/train.py', '--layout', layout]
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"\nError training on {layout}. Exiting.")
            return
    
    print("\n" + "="*60)
    print("Training completed on all layouts!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Evaluate trained models:")
    print("     python src/evaluate.py --all")
    print("  2. Verify results:")
    print("     python verify_results.py")
    print("  3. Generate plots from existing results:")
    print("     python src/generate_plots.py")

if __name__ == '__main__':
    train_all()

