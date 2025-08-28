import os
import json

from src.Include.wrapper.llama_wrapper import LlamaCPP

class Benchmark:
    test_prompt = "Hello " * 10
    max_tokens = 50
    batch_sizes = [4, 8, 16, 32, 64, 128]

    def _save_config(key, value):
        if os.path.exists('device_config.json'):
            with open('device_config.json', "r") as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    config = {}
        else:
            config = {}

        config[key] = value
        with open('device_config.json', "w") as f:
            json.dump(config, f)

    def config_cpu_optimal_batchsize() -> None:
        print("Running CPU benchmark...", flush=True)

        def test_batchsize(batch_size: int) -> int:
            try:
                llama = LlamaCPP(gpu_optimal_batchsize = 0, cpu_optimal_batchsize = batch_size, gpu_acceleration = False)
                throughput = llama.run_inference(Benchmark.test_prompt, Benchmark.max_tokens)
            except Exception as e:
                raise RuntimeError(f"Error during CPU benchmark: {e}")
            finally:
                if llama is not None:
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

        Benchmark._save_config("cpu_optimal_batchsize", best_batchsize)

    def config_gpu_optimal_batchsize() -> None:
        print("Running GPU benchmark...", flush=True)

        if not LlamaCPP.supports_gpu_acceleration():
            print("Your system does not support GPU acceleration")
            Benchmark._save_config("gpu_optimal_batchsize", 0)

            return

        def test_batchsize(batch_size: int) -> int:
            try:
                llama = LlamaCPP(gpu_optimal_batchsize = batch_size, cpu_optimal_batchsize = 0, gpu_acceleration = True)
                throughput = llama.run_inference(Benchmark.test_prompt, Benchmark.max_tokens)
            except Exception as e:
                raise RuntimeError(f"Error during GPU benchmark: {e}")
            finally:
                if llama:
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

        Benchmark._save_config("gpu_optimal_batchsize", best_batchsize)