"""
Utility functions for training, evaluation, and visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from pathlib import Path


def moving_average(data, window=100):
    """Compute moving average of data."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_curves(results_dict, save_path=None, title="Training Progress"):
    """
    Plot training curves for multiple layouts.
    
    Args:
        results_dict: Dictionary mapping layout names to lists of episode rewards
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for layout_name, rewards in results_dict.items():
        if len(rewards) > 0:
            # Compute moving average
            ma_window = min(100, len(rewards) // 10)
            if ma_window > 0:
                smoothed = moving_average(rewards, ma_window)
                episodes = np.arange(len(smoothed)) + ma_window
                plt.plot(episodes, smoothed, label=f'{layout_name} (smoothed)', linewidth=2)
            
            # Plot raw data with transparency
            plt.plot(rewards, alpha=0.3, linewidth=0.5)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Soups Delivered', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_evaluation_results(results_dict, save_path=None, title="Evaluation Results"):
    """
    Plot evaluation results for multiple layouts.
    
    Args:
        results_dict: Dictionary mapping layout names to lists of episode rewards
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for layout_name, rewards in results_dict.items():
        episodes = np.arange(len(rewards))
        plt.plot(episodes, rewards, label=layout_name, marker='o', markersize=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Soups Delivered', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=7, color='r', linestyle='--', label='Target (7)', linewidth=2)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_metrics(metrics_dict, save_path=None, title="Auxiliary Metrics"):
    """
    Plot auxiliary metrics (e.g., onion pickups, dish pickups, etc.).
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    # Create subplots for better visualization
    n_metrics = len([v for v in metrics_dict.values() if len(v) > 0])
    if n_metrics > 0:
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            if len(values) > 0:
                plt.subplot(n_rows, n_cols, idx + 1)
                
                # Compute moving average
                ma_window = min(100, len(values) // 10)
                if ma_window > 0 and len(values) > ma_window:
                    smoothed = moving_average(values, ma_window)
                    episodes = np.arange(len(smoothed)) + ma_window
                    plt.plot(episodes, smoothed, label='Smoothed', linewidth=2, color='blue')
                    # Also show raw data with transparency
                    plt.plot(values, alpha=0.2, color='blue', linewidth=0.5)
                else:
                    plt.plot(values, label=metric_name, linewidth=1, color='blue')
                
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Count', fontsize=10)
                plt.title(metric_name.replace('_', ' ').title(), fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=9)
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def save_results(results, filepath):
    """Save results dictionary to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load results dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class MetricsTracker:
    """Track various metrics during training."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, episode, info):
        """Update metrics from episode info."""
        if isinstance(info, dict):
            # Track soups delivered
            if 'episode' in info and 'soup_delivered' in info['episode']:
                soups = info['episode']['soup_delivered']
                self.metrics['soups_delivered'].append(soups)
            
            # Track other metrics from info dict
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.metrics[key].append(value)
                elif isinstance(value, dict) and 'episode' in value:
                    for ep_key, ep_value in value['episode'].items():
                        if isinstance(ep_value, (int, float)):
                            self.metrics[ep_key].append(ep_value)

