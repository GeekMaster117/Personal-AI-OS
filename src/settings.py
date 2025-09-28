from datetime import timedelta

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

usagedata_dir: str = "data"
sql_dir: str = "sql"
schema_dir: str = sql_dir + "/schema.sql"
fetch_all_apps_titles_dir: str = sql_dir + "/fetch_all_apps_titles.sql"

library_dir: str = "bin"

cache_dir: str = "cache"
model_dir: str = "models/Phi-3-mini-4k-instruct-q4.gguf"

parser_dir: str = library_dir + "/parser"
commands_dir: str = parser_dir + "/commands.bin"
keyword_action_map_dir: str = parser_dir + "/keyword_action_map.bin"
action_pipeline_dir: str = parser_dir + "/action_pipeline.bin"

keyword_argument_maps_dir: str = parser_dir + "/keyword argument maps"
def keyword_argument_map_dir(action: str) -> str:
    return keyword_argument_maps_dir + "/" + action + "/keyword_argument_map.bin"

argument_pipelines_dir: str = parser_dir + "/argument pipelines"
def argument_pipeline_dir(action: str) -> str:
    return argument_pipelines_dir + "/" + action + "/argument_pipeline.bin"

total_model_layers: int = 32
model_window_size: int = 2048
layer_batchsize_weight = 0.5

supported_arch = {61, 75, 86, 89, 120}

max_logs: int = 7
data_limit: int = 3