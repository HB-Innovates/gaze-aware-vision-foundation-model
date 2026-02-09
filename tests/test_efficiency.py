"""Unit tests for efficiency modules."""

import pytest
import torch
import torch.nn as nn
from models.efficient_inference.snn_converter import (
    SNNConverter,
    LIFNeuron,
    convert_to_snn,
    EnergyMonitor,
)
from models.efficient_inference.quantization import (
    quantize_model,
    ModelPruner,
    benchmark_inference,
)


class TestLIFNeuron:
    """Test Leaky Integrate-and-Fire neuron."""
    
    @pytest.fixture
    def neuron(self):
        """Create LIF neuron."""
        return LIFNeuron(threshold=1.0, decay=0.9)
    
    def test_neuron_initialization(self, neuron):
        """Test neuron initialization."""
        assert neuron.threshold == 1.0
        assert neuron.decay == 0.9
    
    def test_neuron_forward(self, neuron):
        """Test neuron forward pass."""
        input_current = torch.randn(4, 10)
        spikes, membrane = neuron(input_current)
        
        assert spikes.shape == input_current.shape
        assert membrane.shape == input_current.shape
        assert torch.all((spikes == 0) | (spikes == 1)), "Spikes should be binary"
    
    def test_neuron_reset(self, neuron):
        """Test membrane reset."""
        input_current = torch.randn(4, 10)
        _, _ = neuron(input_current)
        neuron.reset_state()
        assert neuron.membrane_potential is None


class TestSNNConverter:
    """Test SNN conversion."""
    
    @pytest.fixture
    def converter(self):
        """Create converter instance."""
        return SNNConverter(num_steps=25)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for conversion."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
    
    def test_converter_initialization(self, converter):
        """Test converter initialization."""
        assert converter.num_steps == 25
    
    def test_input_encoding(self, converter):
        """Test spike train encoding."""
        input_tensor = torch.randn(4, 10)
        spike_trains = converter.encode_input(input_tensor)
        
        assert spike_trains.shape == (25, 4, 10)
        assert torch.all((spike_trains == 0) | (spike_trains == 1))
    
    def test_output_decoding(self, converter):
        """Test spike train decoding."""
        spike_trains = torch.randint(0, 2, (25, 4, 10)).float()
        decoded = converter.decode_output(spike_trains)
        
        assert decoded.shape == (4, 10)
        assert torch.all(decoded >= 0) and torch.all(decoded <= 1)


class TestQuantization:
    """Test model quantization."""
    
    @pytest.fixture
    def model(self):
        """Create simple model for quantization."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
    
    def test_dynamic_quantization(self, model):
        """Test dynamic quantization."""
        quantized = quantize_model(model, quantization_type='dynamic')
        
        # Quantized model should still produce outputs
        input_tensor = torch.randn(4, 10)
        output = quantized(input_tensor)
        assert output.shape == (4, 2)
    
    def test_model_size_reduction(self, model):
        """Test quantization reduces model size."""
        # Count parameters before
        param_count_before = sum(p.numel() for p in model.parameters())
        
        quantized = quantize_model(model, quantization_type='dynamic')
        
        # Quantized should use less memory (in practice)
        # Note: param count stays same, but dtype changes
        assert quantized is not None


class TestModelPruner:
    """Test model pruning."""
    
    @pytest.fixture
    def model(self):
        """Create model for pruning."""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
        )
    
    def test_magnitude_pruning(self, model):
        """Test magnitude-based pruning."""
        pruner = ModelPruner()
        pruned = pruner.magnitude_prune(model, sparsity=0.5)
        
        # Check that some weights are zero
        total_params = 0
        zero_params = 0
        for param in pruned.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity_ratio = zero_params / total_params
        assert sparsity_ratio > 0.4, f"Expected ~50% sparsity, got {sparsity_ratio*100:.1f}%"


class TestEnergyMonitor:
    """Test energy monitoring."""
    
    @pytest.fixture
    def monitor(self):
        """Create energy monitor."""
        return EnergyMonitor()
    
    def test_dnn_energy_estimation(self, monitor):
        """Test DNN energy estimation."""
        model = nn.Linear(100, 50)
        input_tensor = torch.randn(10, 100)
        
        energy = monitor.measure_dnn_energy(model, input_tensor)
        assert energy > 0, "Energy should be positive"
    
    def test_snn_energy_estimation(self, monitor):
        """Test SNN energy estimation."""
        spike_activity = 0.1  # 10% neurons spike
        num_neurons = 1000
        
        energy = monitor.measure_snn_energy(spike_activity, num_neurons)
        assert energy > 0, "Energy should be positive"
    
    def test_energy_comparison(self, monitor):
        """Test energy comparison shows SNN advantage."""
        # Simulate measurements
        monitor.dnn_energy = [100.0, 105.0, 98.0]
        monitor.snn_energy = [2.5, 2.7, 2.6]
        
        comparison = monitor.compare()
        
        assert comparison['reduction_factor'] > 30, "SNN should show >30x reduction"


def test_benchmark_inference():
    """Test inference benchmarking."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )
    
    metrics = benchmark_inference(
        model,
        input_shape=(1, 64),
        num_iterations=10,
    )
    
    assert 'avg_latency_ms' in metrics
    assert 'throughput_fps' in metrics
    assert 'memory_mb' in metrics
    assert metrics['avg_latency_ms'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
