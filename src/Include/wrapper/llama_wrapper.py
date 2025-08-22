import pycuda.autoinit
import pycuda.driver as cuda
import ctypes
import os
import psutil
import threading
import time
import pickle
import contextlib

import settings
from Include.loading_spinner import loading_spinner

class LlamaCPP:
    def __init__(self,
            gpu_optimal_batchsize: int, 
            cpu_optimal_batchsize: int, 
            model_path: str | None = None, 
            main_gpu: int | None = None, 
            window_size: int | None = None,
            threads: int | None = None,
            gpu_layers: int | None = None,
            batch_size: int | None = None,
            gpu_acceleration: bool = True,
            debug: bool = False
        ):
        self.debug = debug
        best_device_info = self._get_device_info(gpu_optimal_batchsize, cpu_optimal_batchsize, gpu = gpu_acceleration)

        if best_device_info['arch'] in settings.supported_arch:
            ctypes.CDLL(settings.llama_library_dir + "/llama.dll")

        import llama_cpp

        self.debug and print("Initialising LLM...", flush=True)

        devnull = open(os.devnull, 'w')
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            self.llm = llama_cpp.Llama(
                model_path = model_path or settings.model_dir,
                main_gpu = main_gpu or best_device_info['idx'],
                n_ctx = window_size or settings.model_window_size,
                n_threads = threads or os.cpu_count(),
                n_gpu_layers = gpu_layers or best_device_info['gpu_layers'],
                n_batch = batch_size or best_device_info['batch_size'],
                verbose = False,
            )

    def _save_sys_cache(self, system_prompt: str, cache_dir: str) -> None:
        self.llm.create_chat_completion(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "OK"}
        ])

        with open(cache_dir, "wb") as file:
            pickle.dump(self.llm.save_state(), file)

    def _load_sys_cache(self, cache_dir: str) -> None:
        with open(cache_dir, "rb") as file:
            self.llm.load_state(pickle.load(file))
    
    def _get_device_info(self, gpu_optimal_batchsize: int, cpu_optimal_batchsize: int, gpu: bool = True) -> dict[str, str | int]:
        best_device_info = None
        if gpu:
            self.debug and print("Checking support for GPU accleration...", flush=True)
            best_device_info = LlamaCPP._get_gpu_info(gpu_optimal_batchsize)

        if not best_device_info:
            self.debug and print("Skipping GPU acceleration.")
            memory = psutil.virtual_memory()
            best_device_info = {
                'idx': -1,
                'free_mem': memory.free / (1024 ** 2),
                'total_mem': memory.total / (1024 ** 2),
                'arch': 'cpu',
                'gpu_layers': 0,
                'batch_size': cpu_optimal_batchsize
            }
        
        return best_device_info

    def _get_optimal_config(free_memory, total_layers, gpu_layers, layer_size, kv_cache, activations_token, gpu_optimal_batchsize) -> dict[str, float | int]:
        # VRAM used by layers = (No.of Layers * Size of each Layer) + KV cache per Layer

        vram_layers = (gpu_layers * layer_size) + kv_cache
        vram_left = free_memory - vram_layers

        #Returns negative infinite score when no free VRAM left

        if vram_left <= 0:
            return {'score': -float('inf'), 'batch_size': 0}

        #Batch size decided by memory = VRAM used by layers / activation per token

        vram_optimal_batchsize = vram_left / activations_token

        #Since we don't want to occupy entire GPU VRAM, we discount it by 20%. Also llama.cpp may round up the batch size leading to more memory consumption

        vram_batchsize = int(vram_optimal_batchsize * 0.8)

        batch_size = min(vram_batchsize, gpu_optimal_batchsize)

        #Score = batch size - ((total layers - layers) * weight), where weight is preference of layers over batch size.

        latency_penalty = (total_layers - gpu_layers) * settings.layer_batchsize_weight
        score = batch_size - latency_penalty
        
        config = {
            'score': score,
            'layers': gpu_layers,
            'batch_size': batch_size
        }

        return config

    def _get_gpu_info(gpu_optimal_batchsize: int) -> dict[str, str | int] | None:
        try:
            best_device_info: dict | None = None
            for idx in range(cuda.Device.count()):
                device = cuda.Device(idx)
                context = device.make_context()
                free_mem, total_mem = cuda.mem_get_info()
                context.pop()

                if best_device_info:
                    if (best_device_info['free_mem'] < free_mem) or (best_device_info['free_mem'] == free_mem and best_device_info['total_mem'] < total_mem):
                        best_device_info = {
                            'idx': idx,
                            'free_mem': free_mem,
                            'total_mem': total_mem
                        }
                else:
                    best_device_info = {
                        'idx': idx,
                        'free_mem': free_mem,
                        'total_mem': total_mem
                    }

            if not best_device_info:
                return
            
            device = cuda.Device(best_device_info['idx'])
            compute_capability = device.compute_capability()
            arch = (compute_capability[0] * 10) + compute_capability[1]
            best_device_info['arch'] = arch

            #Total RAM used = layers * [(total_layers / model_size) + (window_size * KV cache per token per layer) + (batch_size * activations_per_token)]
            #All the calculations below happen in MB

            model_size = os.path.getsize(settings.model_dir) / (1024 ** 2)
            layer_size = settings.total_model_layers / model_size

            kvcache_token_layer = 0.0009765625 #Rule of thumb is 1KB of kv cache per token per layer in float16
            total_kvcache = settings.model_window_size * kvcache_token_layer

            activations_token = 0.00390625 #Rule of thumb is 4KB of activations per token in float16

            #We perform a ternary search to find optimal gpu layers and batch size, we decide them by using a score

            def get_config(gpu_layers: int) -> dict[str, float | int]:
                return LlamaCPP._get_optimal_config(
                    best_device_info['free_mem'], 
                    settings.total_model_layers, 
                    gpu_layers, 
                    layer_size, 
                    total_kvcache, 
                    activations_token, 
                    gpu_optimal_batchsize
                )

            low, high = 0, settings.total_model_layers
            while high - low > 2:
                difference = (high - low) // 3
                mid1 = low + difference
                mid2 = high - difference

                mid1_score, mid2_score = get_config(mid1)['score'], get_config(mid2)['score']

                if mid1_score < mid2_score:
                    low = mid1
                else:
                    high = mid2

            best_config = max([get_config(layers) for layers in range(low, high + 1)], key = lambda x : x['score'])

            best_device_info['gpu_layers'] = best_config['layers']
            best_device_info['batch_size'] = best_config['batch_size']

            return best_device_info
        except:
            return
        
    def handle_sys_cache(self, system_prompt: str, cache_name: str) -> None:
        if not os.path.exists(settings.cache_dir):
            os.makedirs(settings.cache_dir)
        cache_dir: str = os.path.join(settings.cache_dir, cache_name + ".bin")

        if os.path.exists(cache_dir):
            self.debug and print('Loading Cache...', flush=True)
            try:
                self._load_sys_cache(cache_dir)
                return
            except:
                self.debug and print('Cache incompatibility detected, Will try caching again.')
        
        self.debug and print('Caching for future use...', flush=True)
        self._save_sys_cache(system_prompt, cache_dir)
        self._load_sys_cache(cache_dir)

    def chat(self, user_prompt: str) -> None:
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        spinner_flag = {'running': True}
        spinner_thread = threading.Thread(
            target = loading_spinner,
            args = ("Thinking", spinner_flag),
            daemon = True
        )
        spinner_thread.start()

        first_chunk = True
        for chunk in self.llm.create_chat_completion(messages=messages, temperature=0.7, top_p=0.9, stream=True):
            if first_chunk:
                spinner_flag['running'] = False
                spinner_thread.join()
                first_chunk = False
                self.debug and print("LLM: ", end='', flush=True)

            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            self.debug and print(content, end='', flush=True)
        self.debug and print("\n")

    def run_inference(self, test_prompt: str, max_tokens: int) -> int:
        start = time.monotonic()
        response = self.llm.create_completion(prompt = test_prompt, max_tokens = max_tokens, temperature=0.1)
        end = time.monotonic()

        usage = response.get('usage', {})
        completion_tokens = usage.get('completion_tokens', max_tokens)

        #Returns tokens processed per second
        return int(completion_tokens / (end - start))
    
    def supports_gpu_acceleration() -> bool:
        return cuda.Device.count() != 0