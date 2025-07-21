import pycuda.autoinit
import pycuda.driver as cuda
import os
import ctypes
from llama_cpp import Llama

import settings

class LlamaCPP:
    def __init__(self):
        self.best_gpu_info = self._get_gpu_info()

        cuda_library = self._get_cuda_library(self.best_gpu_info['arch'])
        if cuda_library:
            ctypes.CDLL(cuda_library)

        llm = Llama(
            model_path = settings.model_dir,
            main_gpu = self.best_gpu_info['idx'],
            n_ctx = 2048,
            n_threads = os.cpu_count(),
            n_gpu_layers = 30
        )
        output = llm("Q: What is the capital of India?\nA:", max_tokens=32, stop=["\n"])
        print(output)

    def _get_gpu_info(self) -> list[int] | None:
        try:
            best_gpu_info = dict()
            for idx in range(cuda.Device.count()):
                device = cuda.Device(idx)
                context = device.make_context()
                free_mem, total_mem = cuda.mem_get_info()
                context.pop()

                if best_gpu_info:
                    if (best_gpu_info['free_mem'] < free_mem) or (best_gpu_info['free_mem'] == free_mem and best_gpu_info['total_mem'] < total_mem):
                        best_gpu_info = {
                            'idx': idx,
                            'free_mem': free_mem,
                            'total_mem': total_mem
                        }
                else:
                    best_gpu_info = {
                        'idx': idx,
                        'free_mem': free_mem,
                        'total_mem': total_mem
                    }

            if not best_gpu_info:
                return None
            
            compute_capability = cuda.Device(best_gpu_info['idx']).compute_capability()
            arch = (compute_capability[0] * 10) + compute_capability[1]
            best_gpu_info['arch'] = arch

            return best_gpu_info
        except:
            return None
        
    def _get_cuda_library(self, arch: int) -> str | None:
        if not arch:
            print("Unable to detect supported GPU, skipping GPU acceleration")
            return

        for library in os.listdir(settings.cuda_dir):
            if library.split('.')[0].split('_')[-1] == str(arch):
                print("Your computer supports GPU acceleration")
                return f"{settings.cuda_dir}/" + library

        print("Unable to detect supported GPU, skipping GPU acceleration")

test = LlamaCPP()