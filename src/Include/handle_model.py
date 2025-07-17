import pynvml

class ModelHandler:
    def __init__(self):
        self.n_gpu_layers = self.get_optimal_gpu_layers()

    def get_optimal_gpu_layers(self):
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                print("No NVIDIA GPUs detected.")
                return 0

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use the first GPU
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem_info.total / (1024 ** 3)

            print(f"Detected NVIDIA GPU with {vram_gb:.2f} GB VRAM")

            pynvml.nvmlShutdown()

            # Heuristic to determine n_gpu_layers
            if vram_gb >= 10:
                return 60
            elif vram_gb >= 8:
                return 45
            elif vram_gb >= 6:
                return 30
            elif vram_gb >= 4:
                return 20
            else:
                return 0

        except Exception as e:
            print(f"GPU detection failed: {e}")
            return 0