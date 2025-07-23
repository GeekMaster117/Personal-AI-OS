from datetime import timedelta
from pathlib import Path

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

metadata_dir: str = "metadata"
cuda_dir: str = "bin"

model_dir: str = "models/Phi-3-mini-4k-instruct-q4.gguf"
total_model_layers = 32
model_window_size = 4096

experimental_model_name: str = "phi3:mini"

max_logs: int = 7
data_limit: int = 3