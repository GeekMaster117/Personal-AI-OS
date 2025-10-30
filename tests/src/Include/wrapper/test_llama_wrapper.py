from unittest.mock import patch, MagicMock

from Include.wrapper.llama_wrapper import LlamaCPP

@patch('Include.wrapper.llama_wrapper.CUDA_AVAILABLE', True)
@patch('pycuda.driver.Device')
@patch('pycuda.driver.mem_get_info')
def test_get_gpu_info_success(mock_mem_get_info, mock_device):
    # Mock GPU device and context
    mock_context = MagicMock()
    mock_device_instance = MagicMock()
    mock_device_instance.make_context.return_value = mock_context
    mock_device_instance.compute_capability.return_value = (8, 6)
    mock_device.return_value = mock_device_instance
    mock_device.count.return_value = 1
    
    # Mock GPU memory info
    mock_mem_get_info.return_value = (8000, 10000)  # free_mem, total_mem
    
    result = LlamaCPP._get_gpu_info(32, 100.0, 10.0, 1.0)
    
    assert result is not None
    assert result['idx'] == 0
    assert result['free_mem'] == 8000
    assert result['total_mem'] == 10000
    assert result['arch'] == 86
    assert isinstance(result['gpu_layers'], int) and result['gpu_layers'] > 0
    assert isinstance(result['batch_size'], int) and result['batch_size'] > 0

@patch('pycuda.driver.Device', Exception)
def test_get_gpu_info_no_cuda():
    result = LlamaCPP._get_gpu_info(32, 100.0, 10.0, 1.0)
    assert result is None

@patch('psutil.virtual_memory')
def test_get_cpu_info(mock_virtual_memory):
    # Mock system memory info
    mock_memory = MagicMock()
    mock_memory.free = 8000 * 1024 * 1024
    mock_memory.total = 16000 * 1024 * 1024
    mock_virtual_memory.return_value = mock_memory
    
    result = LlamaCPP._get_cpu_info(32, 100.0, 10.0, 1.0)
    
    assert result['idx'] == -1
    assert result['free_mem'] == 8000
    assert result['total_mem'] == 16000
    assert result['arch'] == 'cpu'
    assert result['gpu_layers'] == 0
    assert isinstance(result['batch_size'], int) and result['batch_size'] > 0

@patch('time.monotonic')
@patch('llama_cpp.Llama')
@patch.object(LlamaCPP, '_get_device_info')
def test_run_inference(mock_get_device_info, mock_llama, mock_time):
    mock_time.side_effect = [0, 2]  # 2 seconds elapsed
    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {
        'usage': {
            'completion_tokens': 100
        }
    }
    
    mock_llama.return_value = mock_llm
    mock_get_device_info.return_value = {
        'idx': -1,
        'arch': 'cpu',
        'gpu_layers': 0,
        'batch_size': 32
    }
    
    llm = LlamaCPP(32, 32)
    tokens_per_second = llm.run_inference("test prompt", 100)
    
    assert tokens_per_second == 50  # 100 tokens / 2 seconds

@patch('llama_cpp.Llama')
@patch.object(LlamaCPP, '_get_device_info')
def test_get_token_count(mock_get_device_info, mock_llama):
    mock_llm = MagicMock()
    mock_llm.tokenize.return_value = [1, 2, 3, 4]  # 4 tokens
    
    mock_llama.return_value = mock_llm
    mock_get_device_info.return_value = {
        'idx': -1,
        'arch': 'cpu',
        'gpu_layers': 0,
        'batch_size': 32
    }
    
    llm = LlamaCPP(32, 32)
    token_count = llm.get_token_count("test prompt")
    
    assert token_count == 4
    mock_llm.tokenize.assert_called_once()

@patch('threading.Thread')
@patch('llama_cpp.Llama')
@patch.object(LlamaCPP, '_get_device_info')
def test_chat(mock_get_device_info, mock_llama, mock_thread):
    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = iter([
        {"choices": [{"text": "Hello"}]},
        {"choices": [{"text": " World"}]}
    ])
    
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance
    
    mock_llama.return_value = mock_llm
    mock_get_device_info.return_value = {
        'idx': -1,
        'arch': 'cpu',
        'gpu_layers': 0,
        'batch_size': 32
    }
    
    llm = LlamaCPP(32, 32)
    llm.chat("system prompt", "user prompt")
    
    mock_llm.create_completion.assert_called_once()
    mock_thread.assert_called_once()
    mock_thread_instance.start.assert_called_once()
    mock_thread_instance.join.assert_called_once()

@patch('Include.wrapper.llama_wrapper.CUDA_AVAILABLE', True)
@patch.object(LlamaCPP, '_get_device_info')
def test_supports_gpu_acceleration_true(mock_get_device_info):
    mock_get_device_info.return_value = {
        'idx': 0,
        'arch': '61'
    }
    assert LlamaCPP.supports_gpu_acceleration() is True

@patch('Include.wrapper.llama_wrapper.CUDA_AVAILABLE', False)
@patch.object(LlamaCPP, '_get_device_info')
def test_supports_gpu_acceleration_false(mock_get_device_info):
    mock_get_device_info.return_value = {
        'idx': -1,
        'arch': 'cpu'
    }
    assert LlamaCPP.supports_gpu_acceleration() is False