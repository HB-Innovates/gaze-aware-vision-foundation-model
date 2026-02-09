"""Model quantization for efficient inference.

Implements INT8 quantization and other compression techniques
for mobile and embedded deployment.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_qat
import copy


def quantize_model(
    model: nn.Module,
    quantization_type: str = 'dynamic',
    calibration_data: torch.Tensor = None,
) -> nn.Module:
    """
    Quantize model to INT8 for efficient inference.
    
    Args:
        model: PyTorch model to quantize
        quantization_type: 'dynamic', 'static', or 'qat'
        calibration_data: Required for static quantization
        
    Returns:
        Quantized model
        
    Example:
        >>> model = GazePredictor()
        >>> quantized = quantize_model(model, quantization_type='dynamic')
        >>> # Model size reduced by 4x, inference 2-3x faster
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    if quantization_type == 'dynamic':
        # Dynamic quantization (easiest, good for linear layers)
        quantized_model = quantize_dynamic(
            model_copy,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
    elif quantization_type == 'static':
        # Static quantization (best accuracy, requires calibration)
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        quantized_model = _quantize_static(model_copy, calibration_data)
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    return quantized_model


def _quantize_static(
    model: nn.Module,
    calibration_data: torch.Tensor,
) -> nn.Module:
    """Perform static quantization with calibration."""
    # Prepare model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with representative data
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            _ = model(batch)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model


class ModelPruner:
    """Prune model weights for compression.
    
    Implements magnitude-based pruning and structured pruning.
    """
    
    @staticmethod
    def magnitude_prune(
        model: nn.Module,
        sparsity: float = 0.5,
    ) -> nn.Module:
        """
        Prune weights with smallest magnitudes.
        
        Args:
            model: Model to prune
            sparsity: Fraction of weights to prune (0 to 1)
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask
        
        return model
    
    @staticmethod
    def structured_prune(
        model: nn.Module,
        prune_ratio: float = 0.3,
    ) -> nn.Module:
        """
        Remove entire filters/neurons based on importance.
        
        Args:
            model: Model to prune
            prune_ratio: Fraction of filters to remove
            
        Returns:
            Pruned model with reduced parameters
        """
        # Simplified structured pruning
        # In practice, use torch.nn.utils.prune
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                num_to_prune = int(num_filters * prune_ratio)
                
                # Compute filter importance (L1 norm)
                importance = torch.sum(
                    torch.abs(module.weight.data.view(num_filters, -1)),
                    dim=1
                )
                
                # Keep most important filters
                _, indices = torch.topk(importance, num_filters - num_to_prune)
                module.weight.data = module.weight.data[indices]
                if module.bias is not None:
                    module.bias.data = module.bias.data[indices]
        
        return model


def benchmark_inference(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
) -> dict:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        num_iterations: Number of iterations
        
    Returns:
        Dictionary with latency, throughput, memory stats
    """
    import time
    import psutil
    
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)  # ms
    
    # Memory usage
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    return {
        'avg_latency_ms': sum(latencies) / len(latencies),
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'throughput_fps': 1000 / (sum(latencies) / len(latencies)),
        'memory_mb': memory_mb,
    }
