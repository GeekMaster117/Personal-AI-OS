import sys

from benchmark import Benchmark

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "invalid"

    if mode == "help":
        print("Usage: python benchmark_cli.py [mode]")
        print("Modes:")
        print("  cpu: Run CPU benchmark")
        print("  gpu: Run GPU benchmark")
    elif mode == "cpu":
        try:
            Benchmark.config_cpu_optimal_batchsize()
        except Exception as e:
            print("Error occurred during CPU benchmark:", e)
    elif mode == "gpu":
        try:
            Benchmark.config_gpu_optimal_batchsize()
        except Exception as e:
            print("Error occurred during GPU benchmark:", e)
    else:
        print("Invalid mode. Use 'help' for usage information.")