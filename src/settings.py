from datetime import timedelta
from pathlib import Path

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

metadata_dir: str = "metadata"
cuda_dir: str = "bin"

model_dir: str = "models/phi-2.Q4_K_M.gguf"
experimental_model_name: str = "phi3:mini"

max_logs: int = 7
data_limit: int = 3