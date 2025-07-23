import pycuda.autoinit
import pycuda.driver as cuda
import os
import ctypes
from llama_cpp import Llama

import settings

class LlamaCPP:
    def __init__(self):
        self.supported_arch = self._get_supported_arch()

        self.best_gpu_info = self._get_gpu_info()

        cuda_library = self._get_cuda_library(self.best_gpu_info['arch'])
        if cuda_library:
            ctypes.CDLL(cuda_library)

            llm = Llama(
                model_path = settings.model_dir,
                main_gpu = self.best_gpu_info['idx'],
                n_ctx = settings.model_window_size,
                n_threads = os.cpu_count(),
                n_gpu_layers = self.best_gpu_info['gpu_layers'],
                n_batch = self.best_gpu_info['batch_size']
            )
            output = llm("Q: What is the capital of India?\nA:", max_tokens=32, stop=["\n"])
            print(output)
        else:
            print("Skipping GPU acceleration...")

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
                return
            
            device = cuda.Device(best_gpu_info['idx'])
            compute_capability = device.compute_capability()
            arch = (compute_capability[0] * 10) + compute_capability[1]
            best_gpu_info['arch'] = arch

            #Total VRAM used = gpu_layers * [(total_layers / model_size) + (window_size * KV cache per token per layer) + (batch_size * activations_per_token)]
            #All the calculations below happen in MB

            gpu_layers = settings.total_model_layers #Since layers are only 32, we can load them all
            best_gpu_info['gpu_layers'] = gpu_layers

            model_size = os.path.getsize(settings.model_dir) / (1024 ** 2)
            size_layer = settings.total_model_layers / model_size

            kvcache_token_layer = 0.0009765625 #Rule of thumb is 1KB of kv cache per token per layer in float16
            total_kvcache = settings.model_window_size * kvcache_token_layer

            activations_token = 0.00390625 #Rule of thumb is 4KB of activations per token in float16

            #Above equation can be written as
            #batch size = ((Free VRAM / gpu_layers) - size per layer - total kv cache) / activation per token

            batch_size = ((best_gpu_info['free_mem'] / gpu_layers) - size_layer - total_kvcache) / activations_token

            #Since we don't want to occupy entire GPU VRAM, we discount it by 20%
            
            max_batch_size = int(batch_size * 0.8)

            #Less powerful GPU cannot handle large batch size.
            #So we may need to decrease batch size even lower
            #Theoretical GPU capability in GFLOPS = cuda cores * clock rate in GHz * 2

            cores_sm = self._get_cores_per_sm(best_gpu_info['arch'])
            if not cores_sm:
                return
            sm_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
            cuda_cores = sm_count * cores_sm

            device = cuda.Device(best_gpu_info['idx'])
            clock_rate_khz = device.get_attribute(cuda.device_attribute.CLOCK_RATE)
            clock_rate_ghz = clock_rate_khz / 1e6

            ideal_gpu_compute = cuda_cores * clock_rate_ghz * 2

            #Since theoretical GPU capability is not possible to achieve, we discount by 50% for LLM processing.

            gpu_compute = ideal_gpu_compute * 0.5

            #batch size = Total GFLOPS / GFLOPS need for 1 token

            compute_token = 25 #Assuming it takes 25 GFLOPS for 1 token

            batch_size = int(gpu_compute // 25)

            #If batch size is more then max_batch_size it will occupy more VRAM, so we cap it.

            if batch_size > max_batch_size:
                best_gpu_info['batch_size'] = max_batch_size
            else:
                best_gpu_info['batch_size'] = batch_size

            return best_gpu_info
        except:
            return
        
    def _get_cores_per_sm(self, arch: int) -> int | None:
        if not arch:
            print("Unable to detect supported GPU")
            return
        
        core_sm_map: dict[int, int] = {
            61: 128,
            75: 64,
            86: 128,
            89: 128,
            120: 128
        }
        if arch not in core_sm_map:
            print("Unsupported gpu architecture detected")
            return

        return core_sm_map[arch]
        
    def _get_cuda_library(self, arch: int) -> str | None:
        if not arch:
            print("Unable to detect supported GPU")
            return
        if arch not in self.supported_arch:
            print("Unsupported gpu architecture detected")
            return

        for library in os.listdir(settings.cuda_dir):
            if library.split('.')[0].split('_')[-1] == str(arch):
                print("Your computer supports GPU acceleration")
                return f"{settings.cuda_dir}/" + library

        print(f"Unable to find library for {arch} architecture")

    def _get_supported_arch(self) -> set[int]:
        supported_arch: set[int] = set()

        for library in os.listdir(settings.cuda_dir):
            filename = library.split('.')[0]

            idx = len(filename) - 1
            while idx >= 0:
                if filename[idx] == '_':
                    break
                idx -= 1

            arch = filename[idx + 1:]
            if not arch.isdigit():
                print(f'Library {library} has unsupported naming convention, skipping library...')

            supported_arch.add(int(arch))

        return supported_arch            

test = LlamaCPP()