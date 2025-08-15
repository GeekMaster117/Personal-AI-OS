from datetime import timedelta
from pathlib import Path

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

usagedata_dir: str = "data"
cuda_dir: str = "bin"
sys_cache_dir: str = cuda_dir + "/sys_cache.bin"

model_dir: str = "models/Phi-3-mini-4k-instruct-q4.gguf"
total_model_layers: int = 32
model_window_size: int = 4096
layer_batchsize_weight = 0.5

experimental_model_name: str = "phi3:mini"

max_logs: int = 7
data_limit: int = 3