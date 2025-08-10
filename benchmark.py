import os
import ctypes

from src.Include.core.llama_wrapper import LlamaCPP 
import settings

class Benchmark:
    test_prompt = "Hello " * 10
    max_tokens = 50

    def run_cpu_benchmark() -> int:
        print("Running CPU benchmark...", flush=True)
        for batch_size in [8, 16, 32, 64, 128]:
            llama = LlamaCPP(0, batch_size, gpu_acceleration = False, use_cache = False)
            print(llama.run_inference(Benchmark.test_prompt, Benchmark.max_tokens))
            del llama
    
print(Benchmark.run_cpu_benchmark())