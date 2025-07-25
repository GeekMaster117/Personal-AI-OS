import pycuda.autoinit
import pycuda.driver as cuda
import os
import threading
import sys
import time
import itertools
import textwrap
import ctypes
from llama_cpp import Llama

from Include.core.suggestion_engine import SuggestionEngine
from Include.core.metadatadb import MetadataDB

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
                n_batch = self.best_gpu_info['batch_size'],
                verbose = False
            )

            db_handler = MetadataDB(settings.metadata_dir)
            suggestion_engine = SuggestionEngine(db_handler)

            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "assistant", "content": self._get_assistant_prompt(suggestion_engine.processed_logs.get())},
                {"role": "user", "content": ""}
            ]

            while True:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'stop']:
                    print("Exiting conversation.")
                    break

                messages[-1]["content"] = user_input

                spinner_flag = {'running': True}
                spinner_thread = threading.Thread(
                    target = self._loading_spinner,
                    args = ("Thinking", spinner_flag,)
                )
                spinner_thread.start()

                first_chunk = True
                for chunk in llm.create_chat_completion(messages=messages, stream=True):
                    if first_chunk:
                        spinner_flag['running'] = False
                        spinner_thread.join()
                        first_chunk = False
                        print("LLM: ", end='', flush=True)

                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    print(content, end='', flush=True)
                print("\n")
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
    
    def _get_assistant_prompt(self, data) -> str:
        return textwrap.dedent(f"""
            Use this data to provide suggestions:
            {data}
        """)

    def _loading_spinner(self, loading_message, flag):
        spinner = itertools.cycle(['|', '/', '-', '\\'])
        while flag['running']:
            sys.stdout.write(f"\r{loading_message}... {next(spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r \r')
        sys.stdout.flush()

test = LlamaCPP()