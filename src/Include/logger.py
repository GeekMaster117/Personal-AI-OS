from datetime import datetime
from pathlib import Path
import json

log_dir = (Path(__file__) / "../../logs.log").resolve()
print(f"Logging to: {log_dir}")

def log_apps(apps):
    timestamp = datetime.now().isoformat()
    line = {
        "timestamp": timestamp,
        "apps": apps
    }

    with open(log_dir, "a", encoding='utf-8') as log_file:
        log_file.write(json.dumps(line) + "\n")