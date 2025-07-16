from datetime import timedelta

tick: timedelta = timedelta(seconds=30)
time_threshold: timedelta = timedelta(minutes=3)

model_name: str = "phi2"
experimental_model_name: str = "phi3:mini"

max_logs: int = 7
llm_data_limit: int = 3