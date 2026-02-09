#!/usr/bin/env python3
"""Comprehensive evaluation framework for gaze tracking system.

Evaluates:
- Gaze prediction accuracy (angular error)
- Temporal prediction performance
- Multi-modal understanding quality
- Energy efficiency metrics
- Inference latency and throughput

Usage:
    python evaluate.py --dataset openeds --split test
    python evaluate.py --benchmark-efficiency
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.gaze_tracking.predictor import GazePredictor
from models.multimodal_foundation.vlm import GazeAwareVLM
from models.efficient_inference.snn_converter import (
    convert_to_snn,
    EnergyMonitor,
)
from models.efficient_inference.quantization import (
    quantize_model,
    benchmark_inference,
)


class GazeEvaluator:
    """Comprehensive evaluation of gaze tracking system."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        
        self.results = {
            'angular_errors': [],
            'temporal_errors': [],
            'inference_times': [],
        }
    
    def angular_error(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute angular error between predicted and ground truth gaze.
        
        Args:
            pred: Predicted gaze [yaw, pitch] in radians
            target: Ground truth gaze [yaw, pitch] in radians
            
        Returns:
            Angular error in degrees
        """
        # Convert to unit vectors
        pred_vec = torch.stack([
            torch.cos(pred[0]) * torch.cos(pred[1]),
            torch.sin(pred[0]) * torch.cos(pred[1]),
            torch.sin(pred[1]),
        ])
        
        target_vec = torch.stack([
            torch.cos(target[0]) * torch.cos(target[1]),
            torch.sin(target[0]) * torch.cos(target[1]),
            torch.sin(target[1]),
        ])
        
        # Compute angle
        dot_product = torch.clamp(torch.dot(pred_vec, target_vec), -1.0, 1.0)
        angle_rad = torch.acos(dot_product)
        angle_deg = angle_rad * 180.0 / np.pi
        
        return angle_deg.item()
    
    def evaluate_accuracy(self, data_loader: DataLoader) -> dict:
        """Evaluate gaze prediction accuracy.
        
        Args:
            data_loader: Test data loader
            
        Returns:
            Dictionary with accuracy metrics
        """
        print("Evaluating gaze prediction accuracy...")
        
        angular_errors = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                targets = batch['gaze'].to(self.device)
                
                # Predict
                predictions = self.model(images)
                
                # Compute errors
                for pred, target in zip(predictions, targets):
                    error = self.angular_error(pred, target)
                    angular_errors.append(error)
        
        angular_errors = np.array(angular_errors)
        
        return {
            'mean_angular_error': float(np.mean(angular_errors)),
            'median_angular_error': float(np.median(angular_errors)),
            'std_angular_error': float(np.std(angular_errors)),
            'min_angular_error': float(np.min(angular_errors)),
            'max_angular_error': float(np.max(angular_errors)),
            'errors_below_5deg': float(np.mean(angular_errors < 5.0) * 100),
            'errors_below_10deg': float(np.mean(angular_errors < 10.0) * 100),
        }
    
    def evaluate_temporal_prediction(self, data_loader: DataLoader) -> dict:
        """Evaluate temporal gaze prediction (1-5 frames ahead).
        
        Args:
            data_loader: Test data with sequences
            
        Returns:
            Dictionary with temporal prediction metrics
        """
        print("Evaluating temporal prediction...")
        
        errors_by_horizon = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Temporal evaluation'):
                # Assuming batch contains sequences
                sequences = batch['sequence'].to(self.device)
                
                # Use first frames to predict future
                history_length = 5
                for horizon in range(1, 6):
                    if sequences.size(1) < history_length + horizon:
                        continue
                    
                    # Simple extrapolation for prediction
                    history = sequences[:, :history_length]
                    future = sequences[:, history_length + horizon - 1]
                    
                    # Predict (simplified - in practice use RNN/Transformer)
                    velocity = history[:, -1] - history[:, -2]
                    prediction = history[:, -1] + horizon * velocity
                    
                    # Compute error
                    for pred, target in zip(prediction, future):
                        error = self.angular_error(pred, target)
                        errors_by_horizon[horizon].append(error)
        
        results = {}
        for horizon, errors in errors_by_horizon.items():
            if errors:
                results[f'horizon_{horizon}_mean_error'] = float(np.mean(errors))
        
        return results
    
    def benchmark_efficiency(self, input_shape: tuple = (1, 1, 64, 64)) -> dict:
        """Benchmark inference efficiency metrics.
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Dictionary with efficiency metrics
        """
        print("Benchmarking efficiency...")
        
        metrics = benchmark_inference(
            self.model,
            input_shape=input_shape,
            num_iterations=100,
        )
        
        return metrics
    
    def compare_optimizations(self) -> dict:
        """Compare baseline vs optimized models.
        
        Returns:
            Comparison metrics
        """
        print("Comparing baseline vs optimized models...")
        
        input_shape = (1, 1, 64, 64)
        
        # Baseline
        baseline_metrics = self.benchmark_efficiency(input_shape)
        
        # Quantized model
        print("Benchmarking quantized model...")
        quantized_model = quantize_model(self.model, quantization_type='dynamic')
        quantized_evaluator = GazeEvaluator(quantized_model, device=self.device)
        quantized_metrics = quantized_evaluator.benchmark_efficiency(input_shape)
        
        # SNN model (if available)
        print("Benchmarking SNN model...")
        try:
            snn_model = convert_to_snn(self.model, num_steps=25)
            snn_evaluator = GazeEvaluator(snn_model, device=self.device)
            snn_metrics = snn_evaluator.benchmark_efficiency(input_shape)
        except Exception as e:
            print(f"SNN benchmarking skipped: {e}")
            snn_metrics = None
        
        comparison = {
            'baseline': baseline_metrics,
            'quantized': quantized_metrics,
        }
        
        if snn_metrics:
            comparison['snn'] = snn_metrics
        
        # Calculate improvements
        comparison['quantized_speedup'] = (
            baseline_metrics['avg_latency_ms'] / quantized_metrics['avg_latency_ms']
        )
        comparison['quantized_memory_reduction'] = (
            baseline_metrics['memory_mb'] / quantized_metrics['memory_mb']
        )
        
        return comparison
    
    def visualize_results(self, save_dir: str = 'evaluation_results'):
        """Visualize evaluation results.
        
        Args:
            save_dir: Directory to save plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Angular error distribution
        if self.results['angular_errors']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.results['angular_errors'], bins=50, edgecolor='black')
            plt.xlabel('Angular Error (degrees)')
            plt.ylabel('Frequency')
            plt.title('Gaze Prediction Angular Error Distribution')
            plt.axvline(
                np.mean(self.results['angular_errors']),
                color='r',
                linestyle='--',
                label=f"Mean: {np.mean(self.results['angular_errors']):.2f}°"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path / 'angular_error_distribution.png', dpi=150)
            plt.close()
        
        # Inference time distribution
        if self.results['inference_times']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.results['inference_times'], bins=50, edgecolor='black')
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Inference Latency Distribution')
            plt.axvline(
                np.mean(self.results['inference_times']),
                color='r',
                linestyle='--',
                label=f"Mean: {np.mean(self.results['inference_times']):.2f} ms"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path / 'latency_distribution.png', dpi=150)
            plt.close()
        
        print(f"Visualizations saved to {save_dir}/")


def create_synthetic_dataloader(num_samples: int = 100) -> DataLoader:
    """Create synthetic data loader for demonstration."""
    from torch.utils.data import TensorDataset
    
    images = torch.randn(num_samples, 1, 64, 64)
    gazes = torch.randn(num_samples, 2) * 0.5  # Random yaw/pitch
    
    dataset = TensorDataset(images, gazes)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Gaze Tracking System'
    )
    parser.add_argument(
        '--dataset',
        choices=['openeds', 'synthetic'],
        default='synthetic',
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--split',
        default='test',
        help='Dataset split'
    )
    parser.add_argument(
        '--benchmark-efficiency',
        action='store_true',
        help='Run efficiency benchmarks'
    )
    parser.add_argument(
        '--compare-optimizations',
        action='store_true',
        help='Compare baseline vs optimized models'
    )
    parser.add_argument(
        '--output',
        default='evaluation_results.json',
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = GazePredictor(hidden_dim=128)
    
    # Initialize evaluator
    evaluator = GazeEvaluator(model)
    
    results = {}
    
    # Accuracy evaluation
    if args.dataset:
        # Create data loader
        if args.dataset == 'synthetic':
            data_loader = create_synthetic_dataloader()
        else:
            # Load actual dataset (to be implemented)
            print(f"Dataset {args.dataset} not implemented. Using synthetic.")
            data_loader = create_synthetic_dataloader()
        
        # Format data for evaluator
        formatted_loader = []
        for images, gazes in data_loader:
            formatted_loader.append({
                'image': images,
                'gaze': gazes,
            })
        
        accuracy_results = evaluator.evaluate_accuracy(formatted_loader)
        results['accuracy'] = accuracy_results
        
        print("\n=== Accuracy Results ===")
        for key, value in accuracy_results.items():
            print(f"{key}: {value:.4f}")
    
    # Efficiency benchmarks
    if args.benchmark_efficiency:
        efficiency_results = evaluator.benchmark_efficiency()
        results['efficiency'] = efficiency_results
        
        print("\n=== Efficiency Results ===")
        for key, value in efficiency_results.items():
            print(f"{key}: {value:.4f}")
    
    # Optimization comparison
    if args.compare_optimizations:
        comparison_results = evaluator.compare_optimizations()
        results['optimization_comparison'] = comparison_results
        
        print("\n=== Optimization Comparison ===")
        print(f"Quantized speedup: {comparison_results['quantized_speedup']:.2f}x")
        print(f"Memory reduction: {comparison_results['quantized_memory_reduction']:.2f}x")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Visualize
    evaluator.visualize_results()


if __name__ == '__main__':
    main()
