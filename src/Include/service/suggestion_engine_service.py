import os
import json
import threading
import sys

from Include.wrapper.llama_wrapper import LlamaCPP

class SuggestionEngineService:
    def __init__(self):
        if not os.path.exists('device_config.json'):
            raise FileNotFoundError("Configuration file 'device_config.json' not found. Please run the benchmark first.")

        cpu_optimal_batchsize = None
        gpu_optimal_batchsize = None
        with open('device_config.json', 'r') as file:
            device_config = json.load(file)
            try:
                cpu_optimal_batchsize: int = device_config["cpu_optimal_batchsize"]
                gpu_optimal_batchsize: int = device_config["gpu_optimal_batchsize"]

                if not isinstance(cpu_optimal_batchsize, int):
                    raise ValueError("cpu_optimal_batchsize must be an integer.")
                if not isinstance(gpu_optimal_batchsize, int):
                    raise ValueError("gpu_optimal_batchsize must be an integer.")
            except:
                raise ValueError("cpu_optimal_batchsize and gpu_optimal_batchsize not found in the configuration file. Please run the benchmark.")
        
        self._llama: LlamaCPP | None = None
        self._llama_ready: threading.Event = threading.Event()

        thread = threading.Thread(target=self._initialize_llama, args=(cpu_optimal_batchsize, gpu_optimal_batchsize), daemon=True)
        thread.start()

    def _initialize_llama(self, cpu_optimal_batchsize: int, gpu_optimal_batchsize: int):
        class MainOnlyStdout:
            def write(self, text):
                current = threading.current_thread()
                if not current.daemon:
                    sys.__stdout__.write(text)

            def flush(self):
                current = threading.current_thread()
                if not current.daemon:
                    sys.__stdout__.flush()

        original_stdout = sys.stdout
        sys.stdout = MainOnlyStdout()

        self._llama = LlamaCPP(cpu_optimal_batchsize, gpu_optimal_batchsize)

        self._llama_ready.set()