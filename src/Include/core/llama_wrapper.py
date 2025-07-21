import pycuda.autoinit
import pycuda.driver as cuda
import os
import ctypes
from llama_cpp import Llama

import settings

class LlamaCPP:
    def __init__(self):
        arch = self._get_nvidia_gpu_arch()
        cuda_library = self._get_cuda_library(arch)
        if cuda_library:
            ctypes.CDLL(cuda_library)
        print('Running tests...')
        print('GPU acceleration successfully works.')
        
        llm = Llama(
            model_path="models/phi-2.Q4_K_M.gguf",  # <-- Update path to your model
            n_ctx=512,
            n_threads=6,  # Or however many threads your CPU can handle
            n_gpu_layers=30
        )
        output = llm("Q: What is the capital of India?\nA:", max_tokens=32, stop=["\n"])
        print(output)

    def _get_nvidia_gpu_arch(self) -> int | None:
        try:
            if cuda.Device.count() < 1:
                return None
            
            device = cuda.Device(0).compute_capability()
            return (device[0] * 10) + device[1]
        except:
            return None
        
    def _get_cuda_library(self, arch) -> str | None:
        if not arch:
            print("Unable to detect supported GPU, skipping GPU acceleration")
            return

        for library in os.listdir(settings.cuda_dir):
            if library.split('.')[0].split('_')[-1] == str(arch):
                print("Your computer supports GPU acceleration")
                return f"{settings.cuda_dir}/" + library

        print("Unable to detect supported GPU, skipping GPU acceleration")

test = LlamaCPP()