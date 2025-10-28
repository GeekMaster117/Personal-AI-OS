import warnings

CUDA_AVAILABLE = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pycuda.driver as cuda
        cuda.init() # type: ignore If any error occurs, cuda is disabled
        CUDA_AVAILABLE = True
    except:
        CUDA_AVAILABLE = False

import os
import ctypes
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
        best_device_info: dict = LlamaCPP._get_device_info(gpu_optimal_batchsize, cpu_optimal_batchsize, gpu = gpu_acceleration)

        if best_device_info['arch'] in settings.supported_arch:
            ctypes.CDLL(settings.library_dir + "/llama.dll")

        import llama_cpp

        if self.debug:
            print("Initialising LLM...", flush=True)

        devnull = open(os.devnull, 'w')
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            self.llm = llama_cpp.Llama(
                model_path = model_path or settings.model_dir,
                main_gpu = main_gpu or best_device_info['idx'],
                n_ctx = window_size or settings.model_window_size,
                n_threads = threads or os.cpu_count(),
                n_gpu_layers = gpu_layers or best_device_info['gpu_layers'],
                n_batch = batch_size or best_device_info['batch_size'],
                verbose = False
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

    @staticmethod
    def _get_device_info(gpu_optimal_batchsize: int, cpu_optimal_batchsize: int, gpu: bool = True, debug: bool = False) -> dict[str, str | int]:
        #Total RAM used = layers * [(total_layers / model_size) + (window_size * KV cache per token per layer) + (batch_size * activations_per_token)]
        #All the calculations below happen in MB

        model_size = os.path.getsize(settings.model_dir) / (1024 ** 2)
        layer_size = model_size / settings.total_model_layers

        kvcache_token_layer = 0.0009765625 #Rule of thumb is 1KB of kv cache per token per layer in float16
        total_kvcache = settings.model_window_size * kvcache_token_layer

        activations_token = 0.00390625 #Rule of thumb is 4KB of activations per token in float16

        best_device_info = None
        if CUDA_AVAILABLE and gpu:
            if debug:
                print("Checking support for GPU accleration...", flush=True)
            
            best_device_info = LlamaCPP._get_gpu_info(gpu_optimal_batchsize, layer_size, total_kvcache, activations_token)
            if best_device_info is not None and best_device_info["batch_size"] == 0:
                if debug:
                    print("Insufficient GPU memory.")
                
                best_device_info = None

        if not best_device_info:
            if debug:
                print("Skipping GPU acceleration.")
            
            best_device_info = LlamaCPP._get_cpu_info(cpu_optimal_batchsize, layer_size, total_kvcache, activations_token)
        
        if best_device_info['batch_size'] == 0:
            raise RuntimeError("Not enough memory to load the model. Please try closing other applications.")
        return best_device_info

    @staticmethod
    def _get_optimal_config(free_memory, total_layers, layers_loaded, layer_size, total_kv_cache, activations_token, compute_optimal_batchsize) -> dict[str, float | int]:
        # Memory used by layers = (No.of Layers * Size of each Layer) + KV cache per Layer

        memory_layers = (layers_loaded * layer_size) + total_kv_cache
        memory_left = free_memory - memory_layers

        #Returns negative infinite score when no free memory left

        if memory_left < activations_token:
            return {'score': -float('inf'), 'batch_size': 0}

        #Batch size decided by memory = Memory used by layers / activation per token

        memory_full_batchsize = memory_left / activations_token

        #Since we don't want to occupy entire memory, we discount it by 20%. Also llama.cpp may round up the batch size leading to more memory consumption
        #If memory full batchsize equal to 1, then we don't discount it.

        memory_optimal_batchsize = max(1, int(memory_full_batchsize * 0.8))

        batch_size = min(memory_optimal_batchsize, compute_optimal_batchsize)

        #Score = batch size - ((total layers - layers) * weight), where weight is preference of layers over batch size.

        latency_penalty = (total_layers - layers_loaded) * settings.layer_batchsize_weight
        score = batch_size - latency_penalty
        
        config = {
            'score': score,
            'layers': layers_loaded,
            'batch_size': batch_size
        }

        return config

    @staticmethod
    def _get_gpu_info(gpu_optimal_batchsize: int, layer_size: float, total_kvcache: float, activations_token: float) -> dict[str, str | int] | None:
        try:
            best_gpu_info: dict | None = None
            for idx in range(cuda.Device.count()): # type: ignore If any error occurs, returns None
                device = cuda.Device(idx) # type: ignore If any error occurs, returns None
                context = device.make_context()
                free_mem, total_mem = cuda.mem_get_info() # type: ignore If any error occurs, returns None
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
            
            device = cuda.Device(best_gpu_info['idx']) # type: ignore If any error occurs, returns None
            compute_capability = device.compute_capability()
            arch = (compute_capability[0] * 10) + compute_capability[1]
            best_gpu_info['arch'] = arch

            #We perform a ternary search to find optimal gpu layers and batch size, we decide them by using a score

            def get_config(gpu_layers: int) -> dict[str, float | int]:
                return LlamaCPP._get_optimal_config(
                    best_gpu_info['free_mem'], 
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

            best_gpu_info['gpu_layers'] = best_config['layers']
            best_gpu_info['batch_size'] = best_config['batch_size']

            return best_gpu_info
        except:
            return None

    @staticmethod
    def _get_cpu_info(cpu_optimal_batchsize: int, layer_size: float, total_kvcache: float, activations_token: float) -> dict[str, str | int]:
        memory = psutil.virtual_memory()
        free_mem = memory.free / (1024 ** 2)
        total_mem = memory.total / (1024 ** 2)

        cpu_info = {
            'idx': -1,
            'free_mem': free_mem,
            'total_mem': total_mem,
            'arch': 'cpu'
        }

        config = LlamaCPP._get_optimal_config(
            free_mem, 
            settings.total_model_layers, 
            settings.total_model_layers, 
            layer_size, 
            total_kvcache, 
            activations_token, 
            cpu_optimal_batchsize
        )

        cpu_info['gpu_layers'] = 0
        cpu_info['batch_size'] = config['batch_size']

        return cpu_info

    def handle_sys_cache(self, system_prompt: str, cache_name: str) -> None:
        if not os.path.exists(settings.cache_dir):
            os.makedirs(settings.cache_dir)
        cache_dir: str = os.path.join(settings.cache_dir, cache_name + ".bin")

        if os.path.exists(cache_dir):
            if self.debug: 
                print('Loading Cache...', flush=True)

            try:
                self._load_sys_cache(cache_dir)
                return
            except:
                if self.debug:
                    print('Cache incompatibility detected, Will try caching again.')
        
        if self.debug:
            print('Caching for future use...', flush=True)

        self._save_sys_cache(system_prompt, cache_dir)
        self._load_sys_cache(cache_dir)

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 128, stop: list[str] = [], temperature: float = 0.7, top_p: float = 0.9) -> None:
        full_prompt = f"<|system|>{system_prompt}<|end|>\n<|user|>{user_prompt}<|end|>\n<|assistant|>"

        spinner_flag = {'running': True}
        spinner_thread = threading.Thread(
            target = loading_spinner,
            args = ("Thinking", spinner_flag),
            daemon = True
        )
        spinner_thread.start()

        first_chunk = True
        for chunk in self.llm.create_completion(
            prompt = full_prompt,
            max_tokens = max_tokens,
            stop = stop,
            temperature = temperature,
            top_p = top_p, 
            stream = True):
            if first_chunk:
                spinner_flag['running'] = False
                spinner_thread.join()
                first_chunk = False
                print("LLM: ", end='', flush=True)

            content = chunk["choices"]['text'] # type: ignore Chunk is CreateCompletionResponse, Pylance treating it like string
            print(content, end='', flush=True)
        print()

    def run_inference(self, test_prompt: str, max_tokens: int) -> int:
        start = time.monotonic()
        response = self.llm.create_completion(prompt = test_prompt, max_tokens = max_tokens, temperature=0.1)
        end = time.monotonic()

        usage = response.get('usage', {}) # type: ignore Respone is CreateCompletionResponse, Pylance treating it like Iterate[CreateCompletionStreamResponse]
        completion_tokens = usage.get('completion_tokens', max_tokens)

        #Returns tokens processed per second
        return int(completion_tokens / (end - start))
    
    def get_token_count(self, prompt: str) -> int:
        return len(self.llm.tokenize(prompt.encode('utf-8')))
    
    @staticmethod
    def supports_gpu_acceleration() -> bool:
        best_device_info = LlamaCPP._get_device_info(1, 1, gpu = True, debug = False)
        return best_device_info['arch'] != 'cpu'