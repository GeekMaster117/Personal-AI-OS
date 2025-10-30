import os

from enum import Enum

from datetime import timedelta

# Stage environments
class Environment(Enum):
    PROD = "prod"
    DEV = "dev"

# Supported systems
supported_arch: set[int] = {61, 75, 86, 89, 120}

class SupportedOS(Enum):
    WINDOWS = "Windows"
    LINUX = "Linux"
    MACOS = "Darwin"

# Time settings
tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

# Benchmark configurations
device_config_dir: str = "device_config.json"

# Data directories
usagedata_dir: str = "data"
sql_dir: str = "sql"
schema_dir: str = os.path.join(sql_dir, "schema.sql")

# Model settings
model_dir: str = os.path.join("models", "Phi-3-mini-4k-instruct-q4.gguf")

total_model_layers: int = 32
model_window_size: int = 2048

# Library directories
library_dir: str = "bin"

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

# Weight settings
class_day_historical_weight: float = 0.6

layer_batchsize_weight: float = 0.5

# Suggestion settings
max_logs: int = 7
data_limit: int = 3