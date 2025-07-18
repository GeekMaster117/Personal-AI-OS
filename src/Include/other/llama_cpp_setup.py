import pynvml
import subprocess

class LlamaCPPSetup:
    def __init__(self):
        self._init_pynvml()

        self.supports_gpu_acceleration = self._check_gpu_acceleration()

        self._close_pynvml()

    def _init_pynvml(self):
        try:
            pynvml.nvmlInit()
            print("pynvml initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize pynvml: {e}")

    def _close_pynvml(self):
        try:
            pynvml.nvmlShutdown()
            print("pynvml closed successfully.")
        except Exception as e:
            print(f"Failed to close pynvml: {e}")
        
    def _check_gpu_acceleration(self):
        if self._get_nvidia_gpu_count() == 0:
            print("No NVIDIA GPU detected, disabling GPU acceleration.")
            return False
        if not self._check_nvcc():
            print("NVIDIA CUDA compiler (nvcc) not found, disabling GPU acceleration.")
            return False
        if not self._check_cmake():
            print("CMake not found, disabling GPU acceleration.")
            return False
        if not self._check_git():
            print("Git not found, disabling GPU acceleration.")
            return False

        return True

    def _get_nvidia_gpu_count(self):
        try:
            return pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            print(f"Error getting GPU count: {e}")
            return 0
        
    def _check_nvcc(self):
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking nvcc: {e}")
            return False
        
    def _check_cmake(self):
        try:
            result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking cmake: {e}")
            return False
        
    def _check_git(self):
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking git: {e}")
            return False

        
    def check_gpu_acceleration(self):
        return self.supports_gpu_acceleration
    
    def build_llama_cpp(self):
        if not self.supports_gpu_acceleration:
            print("GPU acceleration not supported. Skipping build.")
            return
        
