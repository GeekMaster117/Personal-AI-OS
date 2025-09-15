from datetime import timedelta
from pathlib import Path

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

usagedata_dir: str = "data"
sql_dir: str = "sql"
schema_dir: str = sql_dir + "/schema.sql"
fetch_all_apps_titles_dir: str = sql_dir + "/fetch_all_apps_titles.sql"

llama_library_dir: str = "bin"
cache_dir: str = "cache"
model_dir: str = "models/Phi-3-mini-4k-instruct-q4.gguf"

commands_dir: str = "commands.json"
parser_state_dir: str = "parser_state.bin"

total_model_layers: int = 32
model_window_size: int = 2048
layer_batchsize_weight = 0.5

supported_arch = {61, 75, 86, 89, 120}

max_logs: int = 7
data_limit: int = 3