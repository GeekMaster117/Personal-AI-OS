from datetime import timedelta
from pathlib import Path

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

metadata_dir: Path = Path(__file__).resolve().parent / "metadata"

model_name: str = "phi2"
experimental_model_name: str = "phi3:mini"

max_logs: int = 7
data_limit: int = 3