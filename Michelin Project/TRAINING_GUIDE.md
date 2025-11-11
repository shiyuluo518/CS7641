# Training Guide: Complete Training Workflow

## Overview

This guide covers the complete training workflow, including validation, monitoring, and troubleshooting.

## Pre-Training Validation (REQUIRED)

**Before starting the full 8-hour training, ALWAYS run validation:**

```bash
python validate_training.py
```

### What Validation Does:
1. ✅ Verifies GPU is working correctly
2. ✅ Verifies environment initializes correctly
3. ✅ Verifies models initialize correctly
4. ✅ Verifies replay buffer works
5. ✅ Verifies action selection works
6. ✅ Verifies environment step works
7. ✅ Verifies training update works
8. ✅ Runs a short training test (50 episodes, ~5-10 minutes)
9. ✅ Verifies models can be saved

### Expected Output:
- All checks should pass
- Short training test completes successfully
- Model is saved correctly

### Expected Time: ~5-10 minutes

**If validation fails, fix the issues BEFORE running full training to avoid wasting 8 hours.**

### Common Issues:
- GPU not detected → Check `GPU_SETUP.md`
- Environment errors → Check dependencies
- Model errors → Check PyTorch installation

## Starting Training

### Complete Pipeline (Automatic):
```bash
python run_complete_pipeline.py
```
This runs everything automatically: QMIX training, IQL training, evaluation, and plot generation.

### Manual Training:
```bash
# Train QMIX on all layouts
python run_training.py

# Train IQL on all layouts (for comparison)
python src/train_iql.py --layout cramped_room
python src/train_iql.py --layout coordination_ring
python src/train_iql.py --layout counter_circuit_o_1order
```

## Progress Monitoring

### During Training

While training is running, you can check progress with:

```bash
python check_training_progress.py
```

This will show:
- Current progress (% complete)
- Learning metrics (average soups, rewards, loss)
- Learning trends (improving/stable/decreasing)
- Model save status
- Training process status

**Run this periodically (every 1-2 hours) to monitor training.**

### Check Specific Layout

```bash
python check_training_progress.py --layout cramped_room
```

### Check if Training is Running

```bash
python check_training_progress.py --check_processes
```

## Training Progress Logs

Training automatically saves progress every 200 episodes to:
- `logs/{layout}_training_results.json`

This file is updated during training, so you can check it even while training is running.

## Intermediate Checkpoints

Training saves model checkpoints every 500 episodes to:
- `models/{layout}_checkpoint_{episode}.pth`

These allow you to:
- Resume training if interrupted
- Evaluate intermediate models
- Monitor model evolution

## Training Status Indicators

### Good Signs (Training is Working):
- ✅ Progress percentage increasing
- ✅ Soups delivered increasing over time
- ✅ Rewards increasing
- ✅ Loss decreasing
- ✅ Models being saved
- ✅ Training process running

### Warning Signs (Check Training):
- ⚠️ No progress for many episodes
- ⚠️ Soups staying at 0 for many episodes
- ⚠️ Loss not decreasing
- ⚠️ Rewards not increasing
- ⚠️ Training process not running
- ⚠️ No progress updates

## Expected Progress Timeline

### Quick Mode (1000/1500/1500 episodes):

**cramped_room (1000 episodes, ~2 hours):**
- 0-200 episodes: Exploration, low rewards (0-1 soups)
- 200-500 episodes: Learning coordination, rewards increasing (1-3 soups)
- 500-800 episodes: Improving performance (3-5 soups)
- 800-1000 episodes: Stabilizing, final performance (5-7 soups)

**coordination_ring (1500 episodes, ~3 hours):**
- 0-300 episodes: Exploration
- 300-800 episodes: Learning coordination
- 800-1200 episodes: Improving performance
- 1200-1500 episodes: Stabilizing

**counter_circuit_o_1order (1500 episodes, ~3 hours):**
- Similar to coordination_ring

## Troubleshooting

### Training is Not Making Progress

1. **Check if training is running:**
   ```bash
   python check_training_progress.py --check_processes
   ```

2. **Check recent logs:**
   ```bash
   python check_training_progress.py
   ```

3. **Check GPU usage:**
   ```bash
   nvidia-smi
   ```

4. **Check for errors in logs:**
   - Look for error messages in console output
   - Check `logs/{layout}_training_results.json` for anomalies

### Training Stopped/Crashed

1. **Check last checkpoint:**
   - Look for `models/{layout}_checkpoint_*.pth` files
   - Find the highest episode number

2. **Check progress:**
   ```bash
   python check_training_progress.py
   ```

3. **Check system resources:**
   - GPU memory: `nvidia-smi`
   - Disk space: Check if disk is full
   - RAM: Check if out of memory

### Validation Failed

1. **Check GPU:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check environment:**
   ```bash
   python -c "from src.env_wrapper import OvercookedWrapper; env = OvercookedWrapper('cramped_room'); print('OK')"
   ```

3. **Check dependencies:**
   ```bash
   pip list | grep torch
   pip list | grep overcooked
   ```

## Time Estimates

- **Validation**: ~5-10 minutes
- **cramped_room**: ~2 hours (1000 episodes)
- **coordination_ring**: ~3 hours (1500 episodes)
- **counter_circuit_o_1order**: ~3 hours (1500 episodes)
- **Total**: ~8 hours (plus validation time)

## Quick Commands

```bash
# 1. Validate setup (REQUIRED before training)
python validate_training.py

# 2. Start training
python run_training.py

# 3. Monitor progress (run during training)
python check_training_progress.py

# 4. Check specific layout
python check_training_progress.py --layout cramped_room

# 5. Check if training is running
python check_training_progress.py --check_processes
```

## Safety Features

1. **Automatic Progress Saving**: Progress saved every 200 episodes
2. **Checkpoint Saving**: Models saved every 500 episodes
3. **Progress Monitoring**: Can check progress without interrupting training
4. **Validation Script**: Verify setup before long training runs
5. **Error Handling**: Training continues even if some episodes fail

## After Training Completes

1. **Evaluate models:**
   ```bash
   python src/evaluate.py --all
   ```

2. **Compare algorithms:**
   ```bash
   python src/compare_algorithms.py
   ```

3. **Generate plots:**
   ```bash
   python src/generate_plots.py
   ```

4. **Verify results:**
   ```bash
   python verify_results.py
   ```

## Key Files

- `validate_training.py`: Validation script (run before training)
- `check_training_progress.py`: Progress monitoring script (run during training)
- `logs/{layout}_training_results.json`: Progress data (updated every 200 episodes)
- `models/{layout}_checkpoint_{episode}.pth`: Model checkpoints (every 500 episodes)
- `models/{layout}_final.pth`: Final model (saved at end)

## Summary

**Before Training:**
- ✅ Run `validate_training.py` to verify setup
- ✅ Fix any issues before starting full training

**During Training:**
- ✅ Run `check_training_progress.py` every 1-2 hours
- ✅ Monitor progress and learning metrics
- ✅ Verify training is progressing

**After Training:**
- ✅ Evaluate models
- ✅ Generate plots
- ✅ Compare algorithms

This workflow ensures you can:
1. **Verify setup works** before wasting 8 hours
2. **Monitor progress** during training
3. **Detect issues early** before they waste time
