import os
import ctypes

from src.Include.core.llama_wrapper import LlamaCPP 
import settings

class Benchmark:
    test_prompt = "Hello " * 10
    max_tokens = 50
    batch_sizes = [4, 6, 8, 16, 32, 64]

    def get_cpu_optimal_batchsize() -> int:
        print("Running CPU benchmark...", flush=True)

        def test_batchsize(batch_size: int) -> int:
            llama = LlamaCPP(gpu_optimal_batchsize = 0, cpu_optimal_batchsize = batch_size, gpu_acceleration = False, cache = False)
            throughput = llama.run_inference(Benchmark.test_prompt, Benchmark.max_tokens)
            del llama

            return throughput

        best_batchsize = 2
        best_throughput = 0
        for batch_size in Benchmark.batch_sizes:
            print("\nTesting CPU throughput with batch size:", batch_size)
            throughput = test_batchsize(batch_size)

            if throughput > best_throughput:
                best_batchsize = batch_size
                best_throughput = throughput
            else:
                break

        return best_batchsize
    
    def get_gpu_optimal_batchsize() -> int:
        print("Running GPU benchmark...", flush=True)

        if not LlamaCPP.supports_gpu_acceleration():
            print("Your system does not support GPU acceleration")
            return 0

        def test_batchsize(batch_size: int) -> int:
            llama = LlamaCPP(gpu_optimal_batchsize = batch_size, cpu_optimal_batchsize = 0, gpu_acceleration = True, cache = False)
            throughput = llama.run_inference(Benchmark.test_prompt, Benchmark.max_tokens)
            del llama

            return throughput
        
        best_batchsize = 2
        best_throughput = 0
        for batch_size in Benchmark.batch_sizes:
            print("\nTesting GPU throughput with batch size:", batch_size)
            throughput = test_batchsize(batch_size)

            if throughput > best_throughput:
                best_batchsize = batch_size
                best_throughput = throughput
            else:
                break

        return best_batchsize
    
print(Benchmark.get_cpu_optimal_batchsize())
print(Benchmark.get_gpu_optimal_batchsize())