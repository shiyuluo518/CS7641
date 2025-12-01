# Training Results and Analysis

This directory contains all training results, analysis, and visualizations.

## Directory Structure

```
results/
├── plots/                    # Generated plot images
│   ├── training_metrics_three_plots.png  # Main 3-metric plot
│   ├── eval_mean_reward.png
│   └── eval_mean_length.png
│
├── runs/                     # TensorBoard logs
│   └── deepracer-v0__ppo_deepracer__*/  # Training run directories
│
├── evaluations/             # Evaluation results (if exists)
│   └── *.json               # Evaluation metrics per track
│
├── demos/                    # Demo videos (if exists)
│   └── *.mp4                # Agent behavior videos
│
├── training_results.json    # Training statistics summary
│
├── TRAINING_REPORT.md       # Comprehensive training report
├── TRAINING_RESULTS_EXPLANATION.md  # Detailed results explanation
├── TRAINING_METRICS.md      # Metrics documentation
├── TRAINING_GUIDE.md        # Training guide
│
├── training_results_summary.py  # Script to generate summary
├── plot_training_metrics.py    # Script to plot 3 metrics
└── verify_outputs.py           # Script to verify outputs
```

## Usage

### Generate Training Summary
```bash
cd results
python training_results_summary.py
```

### Plot Training Metrics
```bash
cd results
python plot_training_metrics.py
```

### Verify Outputs
```bash
cd results
python verify_outputs.py
```

### View TensorBoard
```bash
tensorboard --logdir runs
```

## Notes

- All training outputs are automatically saved here during training
- Evaluation results are saved to `evaluations/` when using `src.utils.evaluate()`
- Demo videos are saved to `demos/` when using `src.utils.demo()`
- TensorBoard logs are saved to `runs/` during training

