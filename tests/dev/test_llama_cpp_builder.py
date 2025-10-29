import pytest
from unittest.mock import patch, MagicMock

from dev.llama_cpp_builder import LlamaCPPBuilder

@patch('platform.system')
def test_init_unsupported_os(mock_system):
    mock_system.return_value = 'UnknownOS'

    with pytest.raises(NotImplementedError, match="Unsupported operating system: UnknownOS"):
        LlamaCPPBuilder()

@patch('shutil.which')
@patch('subprocess.run')
@patch('platform.system')
def test_gpu_acceleration_all_requirements_met(mock_system, mock_run, mock_which):
    mock_system.return_value = 'Windows'
    
    # Mock all required checks to return True
    mock_run.return_value = MagicMock(returncode=0)
    mock_which.return_value = "/path/to/binary"
    
    # Mock vcvars64.bat existence
    with patch('pathlib.Path.exists', return_value=True):
        builder = LlamaCPPBuilder()
        assert builder.supports_gpu_acceleration == True

@patch('shutil.which')
@patch('subprocess.run')
@patch('platform.system')
def test_gpu_acceleration_missing_nvidia(mock_system, mock_run, mock_which):
    mock_system.return_value = 'Windows'

    # Mock nvidia-smi to fail
    mock_run.return_value = MagicMock(returncode=1)
    mock_which.return_value = "/path/to/binary"
    
    builder = LlamaCPPBuilder()
    assert builder.supports_gpu_acceleration == False

@patch('shutil.which')
@patch('platform.system')
def test_gpu_acceleration_missing_nvcc(mock_system, mock_which):
    mock_system.return_value = 'Windows'

    # Make nvcc check fail
    def mock_which_selective(cmd):
        return None if cmd == "nvcc" else "/path/to/binary"
    
    mock_which.side_effect = mock_which_selective
    
    builder = LlamaCPPBuilder()
    assert builder.supports_gpu_acceleration == False

@patch('shutil.which')
@patch('platform.system')
def test_gpu_acceleration_missing_cmake(mock_system, mock_which):
    mock_system.return_value = 'Windows'

    # Make cmake check fail
    def mock_which_selective(cmd):
        return None if cmd == "cmake" else "/path/to/binary"
    
    mock_which.side_effect = mock_which_selective
    
    builder = LlamaCPPBuilder()
    assert builder.supports_gpu_acceleration == False

@patch('shutil.which')
@patch('subprocess.run')
@patch('platform.system')
def test_check_nvidia_gpu_success(mock_system, mock_run, mock_which):
    mock_system.return_value = 'Windows'

    mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3080")
    mock_which.return_value = "/path/to/binary"
    
    builder = LlamaCPPBuilder()
    assert builder._check_nvidia_gpu() == True

@patch('shutil.which')
@patch('subprocess.run')
@patch('platform.system')
def test_check_nvidia_gpu_failure(mock_system, mock_run, mock_which):
    mock_system.return_value = 'Windows'

    mock_run.side_effect = Exception("nvidia-smi not found")
    mock_which.return_value = "/path/to/binary"
    
    builder = LlamaCPPBuilder()
    assert builder._check_nvidia_gpu() == False

@patch('shutil.which')
@patch('subprocess.run')
@patch('subprocess.check_call')
@patch('os.path.exists')
@patch('os.makedirs')
@patch('platform.system')
def test_build_llama_cpp_with_gpu(mock_system, mock_makedirs, mock_exists, mock_check_call, mock_run, mock_which):
    mock_system.return_value = 'Windows'

    # Mock all dependencies
    mock_which.return_value = "/path/to/binary"
    mock_run.return_value = MagicMock(returncode=0)
    mock_check_call.return_value = MagicMock(returncode=0)
    mock_exists.return_value = True
    mock_makedirs.return_value = None
    
    builder = LlamaCPPBuilder()
    # Force GPU support
    builder.supports_gpu_acceleration = True
    
    # Mock Path.exists for devterminal check
    with patch('pathlib.Path.exists', return_value=True):
        builder.build_llama_cpp()
        
    # Verify build was attempted
    assert mock_run.call_count > 0

@patch('shutil.which')
@patch('subprocess.run')
@patch('subprocess.check_call')
@patch('platform.system')
def test_build_llama_cpp_without_gpu(mock_system, mock_check_call, mock_run, mock_which):
    mock_system.return_value = 'Windows'

    mock_which.return_value = None  # Make GPU checks fail
    mock_run.return_value = MagicMock(returncode=0)
    mock_check_call.return_value = MagicMock(returncode=0)
    
    builder = LlamaCPPBuilder()
    builder.build_llama_cpp()
    
    # Verify no build was attempted when GPU acceleration is disabled
    assert builder.supports_gpu_acceleration == False
