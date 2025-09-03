import subprocess
import os
import shutil
import platform
from enum import Enum
from pathlib import Path

class LlamaCPPBuilder:
    class SupportedOS(Enum):
        WINDOWS = "Windows"
        LINUX = "Linux"
        MACOS = "Darwin"

    def __init__(self):
        os_name = platform.system()
        if os_name == "Windows":
            self.os_name = LlamaCPPBuilder.SupportedOS.WINDOWS
        elif os_name == "Linux":
            self.os_name = LlamaCPPBuilder.SupportedOS.LINUX
        elif os_name == "Darwin":
            self.os_name = LlamaCPPBuilder.SupportedOS.MACOS
        else:
            raise NotImplementedError(f"Unsupported operating system: {os_name}")

        self.supports_gpu_acceleration = self._check_gpu_acceleration()
        
    def _check_gpu_acceleration(self) -> bool:
        if not self._check_nvidia_gpu():
            print("No NVIDIA GPU detected, disabling GPU acceleration.")
            return False
        if not self._check_nvcc():
            print("NVIDIA CUDA compiler (nvcc) not found, disabling GPU acceleration.")
            return False
        if not self._check_cmake():
            print("CMake not found, disabling GPU acceleration.")
            return False
        if not self._check_devterminal():
            print("Could not find Developer Terminal, disabling GPU acceleration.")
            return False
        if not self._check_git():
            print("Git not found, disabling GPU acceleration.")
            return False

        print("Your system supports GPU acceleration.")
        return True

    def _check_nvidia_gpu(self) -> bool:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking for NVIDIA GPU: {e}")
            return False
        
    def _check_nvcc(self) -> bool:
        return shutil.which("nvcc") is not None
        
    def _check_cmake(self):
        try:
            result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking cmake: {e}")
            return False
        
    def _check_devterminal(self) -> bool:
        if self.os_name != LlamaCPPBuilder.SupportedOS.WINDOWS:
            return True  # Developer Terminal is only relevant for Windows

        candidates = [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
        ]

        for path in candidates:
            if path.exists():
                self.devterminal_dir = path
                return True

        return False
        
    def _check_git(self) -> bool:
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking git: {e}")
            return False

    def _clone_repo(self, repo_url, repo_dir) -> None:
        if os.path.exists(repo_dir):
            print("Repository already cloned.")
            return
        print("Cloning Repository...")
        subprocess.run(["git", "clone", repo_url])
        
    def check_gpu_acceleration(self) -> bool:
        return self.supports_gpu_acceleration
    
    def build_llama_cpp(self) -> None:
        if not self.supports_gpu_acceleration:
            print("GPU acceleration disabled. Skipping build.")
            return
        
        repo_url = "https://github.com/ggerganov/llama.cpp"
        repo_dir = "llama.cpp"
        build_dir = os.path.join(repo_dir, "build")
        
        self._clone_repo(repo_url, repo_dir)

        os.makedirs(build_dir, exist_ok=True)

        architectures = [61, 75, 86, 89, 120]

        if architectures:
            print(f"Building for GPU arch {', '.join(map(str, architectures))}...")

            cmake_command = [
                    "cmake",
                    "-G", '"Visual Studio 17 2022"',
                    "-DLLAMA_BUILD_SHARED_LIBS=on",
                    "-DLLAMA_CUDA=on",
                    "-DLLAMA_CURL=off",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(map(str, architectures))}",
                    ".."
                ]
        else:
            print("Building for CPU...")

            cmake_command = [
                    "cmake",
                    "-G", '"Visual Studio 17 2022"',
                    "-DLLAMA_BUILD_SHARED_LIBS=on",
                    "-DLLAMA_CUDA=off",
                    "-DLLAMA_CURL=off",
                    "-DCMAKE_BUILD_TYPE=Release",
                    ".."
                ]

        if self.os_name == LlamaCPPBuilder.SupportedOS.WINDOWS:
            cmake_command = f'"{self.devterminal_dir}" && ' + ' '.join(cmake_command)
            subprocess.check_call(cmake_command, cwd=build_dir, shell=True)
        else:
            subprocess.check_call(cmake_command, cwd=build_dir)

        subprocess.check_call(cmake_command, cwd=build_dir, shell=True)

        build_command = [
            "cmake", 
            "--build", 
            ".", 
            "--config", 
            "Release"
        ]

        if self.os_name == LlamaCPPBuilder.SupportedOS.WINDOWS:
            build_command = f'"{self.devterminal_dir}" && ' + ' '.join(build_command)
            subprocess.check_call(cmake_command, cwd=build_dir, shell=True)
        else:
            subprocess.check_call(build_command, cwd=build_dir, shell=True)

llama = LlamaCPPBuilder()
llama.build_llama_cpp()