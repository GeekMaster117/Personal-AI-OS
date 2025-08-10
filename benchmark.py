import os
import ctypes

from src.Include.core.llama_wrapper import LlamaCPP 
import settings

class Benchmark:
    test_prompt = "Hello " * 10
    max_tokens = 50

    def run_cpu_benchmark() -> int:
        print("Running CPU benchmark...", flush=True)

        def test_batchsize(batch_size: int) -> int:
            llama = LlamaCPP(gpu_max_batch_size = 0, cpu_max_batch_size = batch_size, gpu_acceleration = False, use_cache = False)
            throughput = llama.run_inference(Benchmark.test_prompt, Benchmark.max_tokens)
            del llama

            return throughput

        two_powers = [i for i in range(2, 7)]

        best_batchsize = 2
        best_throughput = 0
        for power in two_powers:
            batch_size = 2 ** power
            print("\nTesting CPU throughput with batch size:", batch_size)

            throughput = test_batchsize(2 ** power)
            print("Throughput:", throughput)

            if throughput > best_throughput:
                best_batchsize = batch_size
                best_throughput = throughput

        return best_batchsize