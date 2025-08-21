import itertools
import sys
import time

# Meant to be invoked in a separate thread
def loading_spinner(loading_message: str, flag: dict[str, bool]) -> None:
    if "running" not in flag:
        raise ValueError("flag must contain 'running' key.")

    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while flag['running']:
        sys.stdout.write(f"\r{loading_message}... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r \r')
    sys.stdout.flush()