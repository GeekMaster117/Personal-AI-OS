import os
import sys
import subprocess

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join("dev", "requirements", "python-requirements.txt")])