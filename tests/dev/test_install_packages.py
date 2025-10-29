import subprocess
import pytest
from unittest.mock import patch

from dev import install_packages

@patch('subprocess.check_call')
def test_install_package_success(mock_check_call, monkeypatch):
    # Ensure subprocess.check_call succeeds and env vars are forwarded
    mock_check_call.return_value = 0

    env = ['FOO=BAR']
    package = ['mypkg==1.2']

    result = install_packages.install_package(env, package)

    assert result is True
    mock_check_call.assert_called_once()

    # verify env passed to subprocess contains our variable
    _, kwargs = mock_check_call.call_args
    assert 'env' in kwargs
    assert kwargs['env'].get('FOO') == 'BAR'

@patch('subprocess.check_call')
@patch('time.sleep')
def test_install_package_retry_and_fail(mock_sleep, mock_check_call, monkeypatch):
    # Make subprocess.check_call raise CalledProcessError on every attempt
    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])
    
    mock_check_call.side_effect = raise_error

    env = []
    package = ['failingpkg']

    result = install_packages.install_package(env, package, retries=3)

    assert result is False
    assert mock_check_call.call_count == 3
    # ensure we attempted to sleep between retries (called at least once)
    assert mock_sleep.call_count == 2

def test_extract_packages_success(tmp_path):
    # Create a test requirements file
    requirements_file = tmp_path / "requirements.txt"
    requirements_content = """
PYTHONPATH=src@numpy==1.21.0
simplepkg
ENV1=val1;ENV2=val2@package==1.0;--extra-index-url=url
"""
    requirements_file.write_text(requirements_content.strip())
    
    # Test extraction
    packages = install_packages.extract_packages(str(requirements_file))
    
    assert len(packages) == 3
    
    # Check simple package
    assert packages[1] == ([], ['simplepkg'])
    
    # Check package with env var
    assert packages[0] == (['PYTHONPATH=src'], ['numpy==1.21.0'])
    
    # Check complex case with multiple env vars and package options
    assert packages[2] == (
        ['ENV1=val1', 'ENV2=val2'],
        ['package==1.0', '--extra-index-url=url']
    )

def test_extract_packages_empty_file(tmp_path):
    requirements_file = tmp_path / "empty.txt"
    requirements_file.write_text("")
    
    packages = install_packages.extract_packages(str(requirements_file))
    assert packages == []

def test_extract_packages_missing_file(tmp_path):
    non_existent = tmp_path / "non_existent.txt"
    with pytest.raises(FileNotFoundError, match="not found"):
        install_packages.extract_packages(str(non_existent))

def test_extract_packages_is_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot be a folder"):
        install_packages.extract_packages(str(tmp_path))

# Extra edge-case: ensure install_package accepts env-less single-string package
def test_install_package_accepts_single_string_package(monkeypatch):
    with patch('subprocess.check_call') as mock_check:
        mock_check.return_value = 0
        # pass a single-element list replicating how main passes it
        result = install_packages.install_package([], ['onlythis'])
        assert result is True
        mock_check.assert_called_once()