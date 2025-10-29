import os
import pytest
import requests
from unittest.mock import patch, MagicMock, mock_open

from install import download_model, benchmark
from settings import Environment

# Test cases for download_model function
def test_download_model_existing_file_no_reinstall(monkeypatch):
    # Mock os.path.exists to return True
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    
    # Mock input to return 'n'
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    
    # Call the function
    download_model("http://test.url", "test_path")
    # If we reach here without any error, the test passes as the function should
    # return early when user chooses not to reinstall

def test_download_model_successful_download(monkeypatch):
    # Mock os.path.exists to return False (file doesn't exist)
    monkeypatch.setattr(os.path, "exists", lambda x: False)
    
    # Mock requests.get
    mock_response = MagicMock()
    mock_response.headers = {'Content-Length': '1000'}
    mock_response.iter_content.return_value = [b'chunk'] * 10
    
    with patch('requests.get', return_value=mock_response), \
         patch('builtins.open', mock_open()):
        download_model("http://test.url", "test_path")
        # If we reach here without any error, the download was successful

def test_download_model_network_error_retry(monkeypatch):
    # Mock os.path.exists to return False
    monkeypatch.setattr(os.path, "exists", lambda x: False)
    
    # Mock requests.get to fail twice then succeed
    mock_success = MagicMock()
    mock_success.headers = {'Content-Length': '1000'}
    mock_success.iter_content.return_value = [b'chunk'] * 10
    
    mock_error = MagicMock()
    mock_error.side_effect = requests.exceptions.RequestException("Network error")
    
    with patch('requests.get', side_effect=[mock_error, mock_error, mock_success]), \
         patch('builtins.open', mock_open()), \
         patch('time.sleep'):  # Mock sleep to speed up tests
        download_model("http://test.url", "test_path")
        # Should succeed on third try

def test_download_model_all_retries_failed(monkeypatch):
    # Mock os.path.exists to return False
    monkeypatch.setattr(os.path, "exists", lambda x: False)
    
    # Mock requests.get to always fail
    mock_error = MagicMock()
    mock_error.side_effect = requests.exceptions.RequestException("Network error")
    
    with patch('requests.get', side_effect=mock_error), \
         patch('time.sleep'), \
         pytest.raises(requests.exceptions.RequestException):
        download_model("http://test.url", "test_path", retries=3)

# Test cases for benchmark function
def test_benchmark_prod_environment():
    with patch('subprocess.run') as mock_run:
        benchmark("cpu", Environment.PROD)
        mock_run.assert_called_once_with(["benchmark_cli.exe", "cpu"], check=True)

def test_benchmark_dev_environment():
    with patch('subprocess.run') as mock_run:
        with patch('sys.executable', 'python'):
            benchmark("gpu", Environment.DEV)
            mock_run.assert_called_once_with(['python', "benchmark_cli.py", "gpu"], check=True)