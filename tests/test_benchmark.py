import os
import json
from unittest.mock import patch, MagicMock

from benchmark import Benchmark
import settings

@patch('json.dump')
@patch('os.path.exists')
def test_save_config(mock_exists, mock_dump):
    mock_exists.return_value = False

    Benchmark._save_config("test_key", "test_value")

    args, _ = mock_dump.call_args
    
    assert 'test_key' in args[0]
    assert args[0]['test_key'] == 'test_value'

@patch('benchmark.LlamaCPP')
@patch('json.dump')
def test_config_cpu_optimal_batchsize(mock_dump, mock_llama):
    # Mock LlamaCPP instance and its run_inference method
    mock_instance = MagicMock()
    mock_llama.return_value = mock_instance
    
    # Configure mock to return increasing throughput until batch size 32
    mock_instance.run_inference.side_effect = [10, 20, 30, 40, 35]
    
    # Run the benchmark
    Benchmark.config_cpu_optimal_batchsize()

    args, _ = mock_dump.call_args

    assert 'cpu_optimal_batchsize' in args[0]
    assert args[0]['cpu_optimal_batchsize'] == 32

@patch('benchmark.LlamaCPP')
@patch('json.dump')
def test_config_gpu_optimal_batchsize_supported(mock_dump, mock_llama):
    # Mock LlamaCPP instance and its methods
    mock_instance = MagicMock()
    mock_llama.return_value = mock_instance
    mock_llama.supports_gpu_acceleration.return_value = True
    
    # Configure mock to return increasing throughput until batch size 16
    mock_instance.run_inference.side_effect = [10, 20, 25, 20]
    
    # Run the benchmark
    Benchmark.config_gpu_optimal_batchsize()
    
    args, _ = mock_dump.call_args

    assert 'gpu_optimal_batchsize' in args[0]
    assert args[0]['gpu_optimal_batchsize'] == 16

@patch('benchmark.LlamaCPP')
@patch('json.dump')
def test_config_gpu_optimal_batchsize_unsupported(mock_dump, mock_llama):
    # Mock GPU not being supported
    mock_llama.supports_gpu_acceleration.return_value = False
    
    # Run the benchmark
    Benchmark.config_gpu_optimal_batchsize()
    
    args, _ = mock_dump.call_args

    assert 'gpu_optimal_batchsize' in args[0]
    assert args[0]['gpu_optimal_batchsize'] == 0

@patch('json.load')
@patch('json.dump')
def test_error_handling(mock_dump, mock_load):
    mock_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    
    # Should handle invalid JSON gracefully
    Benchmark._save_config("test_key", "test_value")
    
    args, _ = mock_dump.call_args

    assert 'test_key' in args[0]
    assert args[0]['test_key'] == 'test_value'