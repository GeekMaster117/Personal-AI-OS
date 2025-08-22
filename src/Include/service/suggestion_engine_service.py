import os
import json
import threading
import sys
import time

from Include.wrapper.llama_wrapper import LlamaCPP
from Include.loading_spinner import loading_spinner

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

        self.init_llama_thread = threading.Thread(
            target = self._initialize_llama,
            args = (cpu_optimal_batchsize, gpu_optimal_batchsize),
            daemon = True,
            name = "init_llama"
        )
        self.init_llama_thread.start()

    def _initialize_llama(self, cpu_optimal_batchsize: int, gpu_optimal_batchsize: int):
        self._llama = LlamaCPP(cpu_optimal_batchsize, gpu_optimal_batchsize)

    def __del__(self):
        if self._llama:
            del self._llama

    def wait_until_ready(self) -> None:
        spinner_flag = {"running": True}
        spinner_thread = threading.Thread(
            target = loading_spinner, 
            args = ("Initializing Model", spinner_flag), 
            daemon = True
        )
        spinner_thread.start()

        while self.init_llama_thread.is_alive():
            time.sleep(0.5)

        spinner_flag["running"] = False
        spinner_thread.join()