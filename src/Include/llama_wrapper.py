import pycuda.autoinit
import pycuda.driver as cuda
import ctypes
import os
import psutil
import threading
import sys
import time
import itertools
import textwrap
import pickle
import time
from io import StringIO
from contextlib import redirect_stderr

from Include.suggestion_engine import SuggestionEngine
from Include.usagedata_db import UsagedataDB

import settings

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
            cache: bool = True
        ):
        best_device_info = LlamaCPP._get_device_info(gpu_optimal_batchsize, cpu_optimal_batchsize, gpu = gpu_acceleration)

        if best_device_info['arch'] != 'cpu':
            ctypes.CDLL(LlamaCPP._get_cuda_library(best_device_info['arch']))

        import llama_cpp

        print("Initialising LLM...", flush=True)
        with redirect_stderr(redirect_stderr(StringIO)):
            self.llm = llama_cpp.Llama(
                model_path = model_path if model_path else settings.model_dir,
                main_gpu = main_gpu if main_gpu else best_device_info['idx'],
                n_ctx = window_size if window_size else settings.model_window_size,
                n_threads = threads if threads else os.cpu_count(),
                n_gpu_layers = gpu_layers if gpu_layers else best_device_info['gpu_layers'],
                n_batch = batch_size if batch_size else best_device_info['batch_size'],
                verbose = False,
            )

        if cache:
            self._handle_sys_cache()
            
    def _save_sys_cache(self) -> None:
        self.llm.create_chat_completion(messages=[
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": "OK"}
        ])

        with open(settings.sys_cache_dir, "wb") as file:
            pickle.dump(self.llm.save_state(), file)
        
    def _load_sys_cache(self) -> None:
        with open(settings.sys_cache_dir, "rb") as file:
            self.llm.load_state(pickle.load(file))

    def _handle_sys_cache(self) -> None:
        if os.path.exists(settings.sys_cache_dir):
            print('Loading Cache...', flush=True)
            try:
                self._load_sys_cache()
                return
            except:
                print('Cache incompatibility detected, Will try caching again.')
        
        print('Caching for future use...', flush=True)
        self._save_sys_cache()
        self._load_sys_cache()
    
    def _get_system_prompt(self) -> str:
        return textwrap.dedent(f"""
            You are a Personal AI Meta Operating System that gives user suggestions based on their app data.

            You speak warmly and professionally, like a calm coach clear, human, and judgment-free.

            You can generate suggestions in four categories:
            1. Routine: Recognise hidden or visible patterns in the app data, and provide insights on it.
            2. Personal: Give suggestions to improve personal life based on app data.
            3. Professional: Give suggestions to improve professional life based on app data.
            4. Productivity: Give suggestions to improve productivity based on app data.

            You will be given app data in the following format:
            Date created: 2025-07-23T19:22:45.123456 in the format of YYYY-MM-DDTHH:MM:SS.ffffff
            Top {settings.data_limit} Apps and their Top {settings.data_limit} Titles:

            1. App name1:
            - Total Focus Duration: Total time spent actively on app. Example - 1.5 hours
            - Total Duration: Total time spent on app actively and passively. Example - 2 hours
            - Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourN]. Example - [2PM: 30.7 minutes, 4PM: 30 seconds]

            - 1.1 Title name1:
            -- Total Focus Duration: Total time spent actively on title. Example - 30.6 minutes
            -- Total Duration: Total time spent on title actively and passively. Example - 49.3 minutes
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [2PM: 20.4 minutes]

            Between 1.1 and 1.{settings.data_limit} there will be similar data on titles, analyze them similarly

            - 1.{settings.data_limit} Title name{settings.data_limit}:
            -- Total Focus Duration: Total time spent actively on title. Example - 10 seconds
            -- Total Duration: Total time spent on title actively and passively. Example - 1.5 minutes
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [4PM: 10 seconds]

            Between 1 and {settings.data_limit} there will be similar data on apps, analyze them similarly

            {settings.data_limit}. App name{settings.data_limit}:
            - Total Focus Duration: Total time spent actively on app. Example - 4.3 hours
            - Total Duration: Total time spent on app actively and passively. Example - 6.7 hours
            - Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourN]. Example - [11AM: 2.9 hours, 9AM: 1 hour, 12PM: 20.3 minutes]

            - {settings.data_limit}.1 Title name1:
            -- Total Focus Duration: Total time spent actively on title. Example - 40 minutes
            -- Total Duration: Total time spent on title actively and passively. Example - 1.1 hours
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [9PM: 40 minutes]

            Between {settings.data_limit}.1 and {settings.data_limit}.{settings.data_limit} there will be similar data on titles, analyze them similarly

            - {settings.data_limit}.{settings.data_limit} Title name{settings.data_limit}:
            -- Total Focus Duration: Total time spent actively on title. Example - 2.6 hours
            -- Total Duration: Total time spent on title actively and passively. Example - 5.4 hours
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [9AM: 14.6 minutes, 11AM: 2.9 hours]


            A title is a specific window or file, e.g., "YouTube - Firefox" or "main.py - VSCode".
            An app is a general application (e.g., Firefox, VSCode).
            Hourly Focus Duration are grouped by hour. Each hour contains how much time spent actively in that hour (Hour is represented in 12-hour format)

            Instructions:
            - NEVER assume or hallucinate any missing data.
            - ONLY MENTION the data you see in the app data, if the data you want to mention is not in the app data, do not mention it.
            - If there is not enough data to make a suggestion, say exactly: "Not enough data to make suggestions."
            - You may provide 0 to 2 suggestions from any of the categories, but only if justified by data.
            - Each suggestion must be 1 to 2 concise sentences, and must be only 1 paragraph.
            - Each sentence must be 10 to 20 words long.
            - If you are unsure whether a suggestion is supported, do not provide one for that category.

            You are not just analyzing — you are mentoring gently, like a productivity coach fused into an OS.
        """)

    def _loading_spinner(self, loading_message, flag):
        spinner = itertools.cycle(['|', '/', '-', '\\'])
        while flag['running']:
            sys.stdout.write(f"\r{loading_message}... {next(spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r \r')
        sys.stdout.flush()

    def _get_supported_arch() -> set[int]:
        supported_arch: set[int] = set()

        for library in os.listdir(settings.cuda_dir):
            filename, filetype = library.split('.')

            if filetype != 'dll':
                continue

            idx = len(filename) - 1
            while idx >= 0:
                if filename[idx] == '_':
                    break
                idx -= 1

            arch = filename[idx + 1:]
            if arch.isdigit():
                supported_arch.add(int(arch))

        return supported_arch

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
        
    def _get_device_info(gpu_optimal_batchsize: int, cpu_optimal_batchsize: int, gpu: bool = True) -> dict[str, str | int]:
        best_device_info = None
        if gpu:
            print("Checking support for GPU accleration...", flush=True)
            best_device_info = LlamaCPP._get_gpu_info(gpu_optimal_batchsize)

        if not best_device_info:
            print("Skipping GPU acceleration.")
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
            
    def _get_cuda_library(arch: int) -> str | None:
        if not arch:
            print("Unable to detect supported GPU")
            return
        
        supported_arch = LlamaCPP._get_supported_arch()
        if arch not in supported_arch:
            print("Unsupported gpu architecture detected")
            return

        for library in os.listdir(settings.cuda_dir):
            if library.split('.')[0].split('_')[-1] == str(arch):
                cuda_library = f"{settings.cuda_dir}/" + library
                return cuda_library

        print(f"Unable to find library for {arch} architecture")

    def chat(self, user_prompt: str, suffix: str) -> None:
        messages = [
            {"role": "user", "content": user_prompt + suffix}
        ]

        spinner_flag = {'running': True}
        spinner_thread = threading.Thread(
            target = self._loading_spinner,
            args = ("Thinking", spinner_flag,)
        )
        spinner_thread.start()

        first_chunk = True
        for chunk in self.llm.create_chat_completion(messages=messages, temperature=0.7, top_p=0.9, stream=True):
            if first_chunk:
                spinner_flag['running'] = False
                spinner_thread.join()
                first_chunk = False
                print("LLM: ", end='', flush=True)

            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            print(content, end='', flush=True)
        print("\n")

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

# llama = LlamaCPP()

# db_handler = MetadataDB(settings.metadata_dir)
# suggestion_engine = SuggestionEngine(db_handler)

# def get_suffix() -> str:
#     return textwrap.dedent(f"""
#         Use this data to generate suggestions
#         {suggestion_engine.processed_logs.get()}
#     """)

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ['exit', 'quit', 'stop']:
#         print("Exiting conversation...", flush=True)
#         break

#     llama.chat(user_input, get_suffix())