import time
import socket
import subprocess
import psutil
import requests

class HandleOllama:
    def __init__(self, ipv4: str = "127.0.0.1", port: int = 11434):
        self.ipv4: str = ipv4
        self.port: int = port

    def _is_connection_available(self, ipv4: str, port: int) -> bool:
        try:
            with socket.create_connection((ipv4, port), timeout=2):
                return True
        except (ConnectionRefusedError, socket.timeout):
            return False

    def _kill_process_on_port(self, port: int) -> bool:
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr.port == port:
                        proc.kill()
                        return True
            except Exception:
                continue
        return False

    def is_running(self) -> bool:
        return self._is_connection_available(self.ipv4, self.port)

    def start(self, timeout: int = 10) -> bool:
        if self.is_running():
            print("Ollama is already running.")
            return True

        try:
            start_time: float = time.time()

            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            while not self.is_running():
                if time.time() - start_time > timeout:
                    print("Timeout while waiting for Ollama to start.")
                    return False
                time.sleep(0.5)
        
            print("Ollama started successfully.")
            return True
        except Exception as e:
            print(f"Failed to start Ollama: {e}")
            return False
        
    def ensure_model(self, model_name: str) -> bool:
        try:
            response: requests.Response = requests.get(f"http://{self.ipv4}:{self.port}/api/tags")
            response.raise_for_status()
            models: list[str] = [m["name"] for m in response.json().get("models", [])]

            if model_name in models:
                return True
            else:
                print(f"Model '{model_name}' not found. Pulling from Ollama...")
                subprocess.run(["ollama", "pull", model_name], check=True)
                print(f"Model '{model_name}' installed successfully.")
                return True
        except requests.exceptions.ConnectionError:
            print("Ollama is not running. Please start ollama first.")
        except subprocess.CalledProcessError:
            print(f"Failed to pull model '{model_name}'.")
        except Exception as e:
            print(f"Unexpected error in ensure_model(): {e}")

        return False

    def stop(self) -> bool:
        if not self.is_running():
            print("Ollama is not running.")
            return False

        if self._kill_process_on_port(self.port):
            print("Ollama stopped successfully.")
            return True
        else:
            print("Failed to stop Ollama.")
            return False