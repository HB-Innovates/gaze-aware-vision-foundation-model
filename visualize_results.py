#!/usr/bin/env python3
"""Visualization tools for gaze tracking results.

Generates:
- Gaze heatmaps
- Attention mechanism visualizations
- Temporal prediction trajectories
- Energy efficiency comparisons

Usage:
    python visualize_results.py --input results.json
    python visualize_results.py --create-demo-plots
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import cv2


class ResultsVisualizer:
    """Visualization toolkit for gaze tracking results."""
    
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
    
    def plot_gaze_heatmap(
        self,
        gaze_points: list,
        image_shape: tuple = (1080, 1920),
        save_name: str = 'gaze_heatmap.png',
    ):
        """Create gaze heatmap visualization.
        
        Args:
            gaze_points: List of (x, y) gaze coordinates
            image_shape: (height, width) of display
            save_name: Output filename
        """
        # Create heatmap
        heatmap = np.zeros(image_shape, dtype=np.float32)
        
        for x, y in gaze_points:
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                cv2.circle(
                    heatmap,
                    (int(x), int(y)),
                    radius=50,
                    color=1.0,
                    thickness=-1
                )
        
        # Blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax.set_title('Gaze Attention Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Gaze Density', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to {self.output_dir / save_name}")
    
    def plot_temporal_trajectory(
        self,
        gaze_sequence: list,
        predictions: list = None,
        save_name: str = 'temporal_trajectory.png',
    ):
        """Visualize gaze trajectory over time.
        
        Args:
            gaze_sequence: List of (yaw, pitch) tuples
            predictions: Optional predicted future gaze points
            save_name: Output filename
        """
        gaze_array = np.array(gaze_sequence)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Yaw over time
        axes[0, 0].plot(gaze_array[:, 0], 'b-', linewidth=2, label='Ground Truth')
        if predictions:
            pred_array = np.array(predictions)
            axes[0, 0].plot(
                range(len(gaze_array), len(gaze_array) + len(pred_array)),
                pred_array[:, 0],
                'r--',
                linewidth=2,
                label='Prediction'
            )
        axes[0, 0].set_xlabel('Time (frames)')
        axes[0, 0].set_ylabel('Yaw (degrees)')
        axes[0, 0].set_title('Horizontal Gaze Movement')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pitch over time
        axes[0, 1].plot(gaze_array[:, 1], 'g-', linewidth=2, label='Ground Truth')
        if predictions:
            axes[0, 1].plot(
                range(len(gaze_array), len(gaze_array) + len(pred_array)),
                pred_array[:, 1],
                'r--',
                linewidth=2,
                label='Prediction'
            )
        axes[0, 1].set_xlabel('Time (frames)')
        axes[0, 1].set_ylabel('Pitch (degrees)')
        axes[0, 1].set_title('Vertical Gaze Movement')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D trajectory
        axes[1, 0].plot(gaze_array[:, 0], gaze_array[:, 1], 'b-', linewidth=2, alpha=0.7)
        axes[1, 0].scatter(
            gaze_array[0, 0],
            gaze_array[0, 1],
            c='green',
            s=100,
            label='Start',
            zorder=5
        )
        axes[1, 0].scatter(
            gaze_array[-1, 0],
            gaze_array[-1, 1],
            c='red',
            s=100,
            label='End',
            zorder=5
        )
        if predictions:
            axes[1, 0].plot(
                pred_array[:, 0],
                pred_array[:, 1],
                'r--',
                linewidth=2,
                alpha=0.7,
                label='Predicted'
            )
        axes[1, 0].set_xlabel('Yaw (degrees)')
        axes[1, 0].set_ylabel('Pitch (degrees)')
        axes[1, 0].set_title('2D Gaze Trajectory')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Velocity profile
        velocities = np.sqrt(np.sum(np.diff(gaze_array, axis=0)**2, axis=1))
        axes[1, 1].plot(velocities, 'purple', linewidth=2)
        axes[1, 1].fill_between(range(len(velocities)), velocities, alpha=0.3)
        axes[1, 1].set_xlabel('Time (frames)')
        axes[1, 1].set_ylabel('Gaze Velocity (deg/frame)')
        axes[1, 1].set_title('Gaze Movement Speed')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory plot saved to {self.output_dir / save_name}")
    
    def plot_energy_comparison(
        self,
        energy_data: dict,
        save_name: str = 'energy_comparison.png',
    ):
        """Compare energy consumption across model variants.
        
        Args:
            energy_data: Dict with {model_name: energy_uJ}
            save_name: Output filename
        """
        models = list(energy_data.keys())
        energies = list(energy_data.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, energies, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylabel('Energy Consumption (µJ)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Energy Efficiency Comparison\n(Lower is Better)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_yscale('log')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f} µJ',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        # Add reduction factors
        if 'Baseline DNN' in models and 'SNN' in models:
            baseline_idx = models.index('Baseline DNN')
            snn_idx = models.index('SNN')
            reduction = energies[baseline_idx] / energies[snn_idx]
            
            ax.text(
                0.5,
                0.95,
                f'SNN: {reduction:.1f}x Energy Reduction',
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Energy comparison saved to {self.output_dir / save_name}")
    
    def plot_accuracy_vs_efficiency(
        self,
        results: list,
        save_name: str = 'accuracy_efficiency_tradeoff.png',
    ):
        """Plot accuracy vs efficiency tradeoff.
        
        Args:
            results: List of dicts with 'name', 'accuracy', 'latency', 'energy'
            save_name: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy vs Latency
        for result in results:
            axes[0].scatter(
                result['latency'],
                result['accuracy'],
                s=200,
                alpha=0.7,
                label=result['name']
            )
            axes[0].annotate(
                result['name'],
                (result['latency'], result['accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
        
        axes[0].set_xlabel('Latency (ms)', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Accuracy vs Latency', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Accuracy vs Energy
        for result in results:
            axes[1].scatter(
                result['energy'],
                result['accuracy'],
                s=200,
                alpha=0.7,
                label=result['name']
            )
            axes[1].annotate(
                result['name'],
                (result['energy'], result['accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
        
        axes[1].set_xlabel('Energy (µJ)', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Accuracy vs Energy', fontsize=14, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Tradeoff plot saved to {self.output_dir / save_name}")
    
    def create_demo_plots(self):
        """Generate comprehensive demo plots for interview."""
        print("Creating demonstration plots...")
        
        # 1. Synthetic gaze heatmap
        gaze_points = [
            (960 + np.random.randn()*100, 540 + np.random.randn()*100)
            for _ in range(500)
        ]
        self.plot_gaze_heatmap(gaze_points)
        
        # 2. Temporal trajectory
        t = np.linspace(0, 4*np.pi, 100)
        gaze_sequence = list(zip(
            20 * np.sin(t) + np.random.randn(100)*2,
            15 * np.cos(t) + np.random.randn(100)*2
        ))
        predictions = list(zip(
            20 * np.sin(t[-5:] + 0.5) + np.random.randn(5)*2,
            15 * np.cos(t[-5:] + 0.5) + np.random.randn(5)*2
        ))
        self.plot_temporal_trajectory(gaze_sequence, predictions)
        
        # 3. Energy comparison
        energy_data = {
            'Baseline DNN': 456.8,
            'Quantized INT8': 342.5,
            'SNN': 12.0,  # 38x reduction
        }
        self.plot_energy_comparison(energy_data)
        
        # 4. Accuracy-efficiency tradeoff
        results = [
            {'name': 'Baseline', 'accuracy': 95.2, 'latency': 12.5, 'energy': 456.8},
            {'name': 'Quantized', 'accuracy': 94.8, 'latency': 4.2, 'energy': 342.5},
            {'name': 'SNN', 'accuracy': 94.5, 'latency': 3.1, 'energy': 12.0},
        ]
        self.plot_accuracy_vs_efficiency(results)
        
        print(f"\nAll demo plots created in {self.output_dir}/")
        print("These visualizations are ready for interview presentation!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Gaze Tracking Results'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input JSON file with results'
    )
    parser.add_argument(
        '--create-demo-plots',
        action='store_true',
        help='Create demonstration plots'
    )
    parser.add_argument(
        '--output-dir',
        default='visualizations',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(output_dir=args.output_dir)
    
    if args.create_demo_plots:
        visualizer.create_demo_plots()
    elif args.input:
        # Load and visualize results from file
        with open(args.input) as f:
            results = json.load(f)
        
        # Create visualizations based on available data
        print(f"Loaded results from {args.input}")
        # Implement specific visualizations based on results structure
    else:
        print("Please specify --input or --create-demo-plots")


if __name__ == '__main__':
    main()
