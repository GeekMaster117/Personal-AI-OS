import sys

from benchmark import Benchmark

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "help":
        print("Usage: python benchmark_cli.py [mode]")
        print("Modes:")
        print("  cpu: Run CPU benchmark")
        print("  gpu: Run GPU benchmark")
    elif mode == "cpu":
        Benchmark.config_cpu_optimal_batchsize()
    elif mode == "gpu":
        Benchmark.config_gpu_optimal_batchsize()
    else:
        print("Invalid mode. Use 'help' for usage information.")