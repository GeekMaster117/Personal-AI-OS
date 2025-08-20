import os
import json

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

        self.llama = LlamaCPP(cpu_optimal_batchsize, gpu_optimal_batchsize)

test = SuggestionEngineService()