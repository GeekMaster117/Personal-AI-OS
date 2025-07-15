from datetime import timedelta

tick: timedelta = timedelta(seconds=30)
downtime_buffer: timedelta = timedelta(minutes=3)

model_name: str = "phi3:mini"
llm_data_limit: int = 3