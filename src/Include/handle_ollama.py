import time
import socket
import subprocess
import psutil

class HandleOllama:
    def __init__(self, ipv4="127.0.0.1", port=11434):
        self.ipv4 = ipv4
        self.port = port

    def _is_connection_available(self, ipv4, port):
        try:
            with socket.create_connection((ipv4, port), timeout=2):
                return True
        except (ConnectionRefusedError, socket.timeout):
            return False

    def _kill_process_on_port(self, port):
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr.port == port:
                        proc.kill()
                    return True
            except Exception:
                continue
        return False

    def is_running(self):
        return self._is_connection_available(self.ipv4, self.port)

    def start(self, timeout=10):
        if self.is_running():
            print("Ollama is already running.")
            return True

        try:
            start_time = time.time()

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

    def stop(self):
        if not self.is_running():
            print("Ollama is not running.")
            return False

        if self._kill_process_on_port(self.port):
            print("Ollama stopped successfully.")
            return True
        else:
            print("Failed to stop Ollama.")
            return False