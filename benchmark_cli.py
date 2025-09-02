import sys
import textwrap
import random
import time
import sys

from benchmark import Benchmark

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "invalid"

    if mode == "help":
        print("Usage: python benchmark_cli.py [mode]")
        print("Modes:")
        print("  cpu: Run CPU benchmark")
        print("  gpu: Run GPU benchmark")
        print()
        print("Oh!! You're actually exploring this deep, your curiosity for being a 'GEEK' won't go unrewarded")
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
    elif mode == "GEEK":
        # ANSI color codes
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        RESET = "\033[0m"

        def slow_print(text, delay=0.02, color=RESET):
            for c in text:
                sys.stdout.write(color + c + RESET)
                sys.stdout.flush()
                time.sleep(delay)
            print()

        slow_print("Initializing secret protocol...", 0.04, CYAN)
        time.sleep(0.5)
        slow_print("Calibrating quantum cores...", 0.04, YELLOW)
        time.sleep(0.5)
        slow_print("Activating Turbo Benchmark Mode!", 0.04, GREEN)
        time.sleep(0.8)
        print()

        for i in range(10):
            score = random.randint(50000, 999999)
            sys.stdout.write(f"\r{CYAN}Benchmark score: {score} GFLOPS{RESET}   ")
            sys.stdout.flush()
            time.sleep(0.2)
        print()

        time.sleep(0.5)
        print(GREEN + f"\nFinal Geek Score: {random.randint(1000000, 2000000)} GFLOPS 🚀" + RESET)
        time.sleep(0.5)

        banner = textwrap.dedent("""
         ██████╗ ███████╗███████╗██╗  ██╗███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗  ██╗ ██╗███████╗
        ██╔════╝ ██╔════╝██╔════╝██║ ██╔╝████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗███║███║╚════██║
        ██║  ███╗█████╗  █████╗  █████╔╝ ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝╚██║╚██║    ██╔╝
        ██║   ██║██╔══╝  ██╔══╝  ██╔═██╗ ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗ ██║ ██║   ██╔╝ 
        ╚██████╔╝███████╗███████╗██║  ██╗██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║ ██║ ██║   ██║  
         ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═╝ ╚═╝   ╚═╝
        """)
        print(banner)
    else:
        print("Invalid mode. Use 'help' for usage information.")