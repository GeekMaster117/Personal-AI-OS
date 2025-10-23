import os

from datetime import timedelta

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

usagedata_dir: str = "data"
sql_dir: str = "sql"
schema_dir: str = os.path.join(sql_dir, "schema.sql")
fetch_all_apps_titles_dir: str = os.path.join(sql_dir, "fetch_all_apps_titles.sql")

library_dir: str = "bin"

cache_dir: str = "cache"
model_dir: str = "models/Phi-3-mini-4k-instruct-q4.gguf"

parser_dir: str = os.path.join(library_dir, "parser")

commands_dir: str = os.path.join(parser_dir, "commands.bin")
keyword_action_map_dir: str = os.path.join(parser_dir, "keyword_action_map.bin")
action_pipeline_dir: str = os.path.join(parser_dir, "action_pipeline.bin")
def keyword_argument_map_dir(action: str) -> str:
    return os.path.join(parser_dir, action, "keyword_argument_map.bin")

def argument_pipeline_dir(action: str) -> str:
    return os.path.join(parser_dir, action, "argument_pipeline.bin")

parser_executable_dir: str = os.path.join(library_dir, "parser executable")

app_executablepath_map_dir: str = os.path.join(parser_executable_dir, "app_executablepath_map.bin")
nickname_app_map_dir: str = os.path.join(parser_executable_dir, "nickname_app_map.bin")
class_app_map_dir: str = os.path.join(parser_executable_dir, "class_app_map.bin")

class_day_historical_weight: float = 0.6

total_model_layers: int = 32
model_window_size: int = 2048
layer_batchsize_weight: float = 0.5

supported_arch: set[int] = {61, 75, 86, 89, 120}

max_logs: int = 7
data_limit: int = 3