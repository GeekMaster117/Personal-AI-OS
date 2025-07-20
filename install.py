import os
import requests
import sys
import subprocess
import time

def install_package(env: list[str], package: list[str], retries: int = 3, delay: int = 2):
    global_env = os.environ.copy()
    for variable in env:
        key, value = variable.split('=')
        global_env[key] = value

    for attempt in range(1, retries + 1):
        try:
            print(f"Installing: {package[-1]}")

            subprocess.check_call([sys.executable, "-m", "pip", "install" , *package], env=global_env)

            print(f"Installed: {package[-1]}\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error on attempt {attempt}: {e}")
            if attempt < retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Install failed after multiple attempts.\n")
                raise

    print(f"Failed to install {package} after {retries} attempts.\n")
    return False

def download_model(url: str, dest_path: str, retries: int = 3, timeout: int = 30):
    if os.path.exists(dest_path):
        print(f"Model already exists at {dest_path}\n")
        return

    print("Downloading model from:", url)

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get('Content-Length', 0))
                downloaded = 0
                start = time.time()

                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            percent = (downloaded / total) * 100 if total else 0
                            done = int(50 * downloaded / total) if total else 0
                            sys.stdout.write(f"\r[{'=' * done}{'.' * (50 - done)}] {percent:.2f}%")
                            sys.stdout.flush()
            print("\nDownload completed.\n")
            return
        except requests.exceptions.RequestException as e:
            print(f"Network error on attempt {attempt}: {e}")
            if attempt < retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Download failed after multiple attempts.\n")
                if os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except Exception as cleanup_error:
                        print(f"Failed to delete incomplete file: {dest_path} due to {cleanup_error}. Please delete it manually, before trying to install again.\n")
                raise

if __name__ == "__main__":
    with open("requirements/python-requirements.txt", "r") as file:
        for line in file:
            env_package = line.strip().split('@')
            if not env_package:
                continue
            
            if len(env_package) == 1:
                install_package([], env_package)
            else:
                env, package = env_package[0].split(';'), env_package[1].split(';')
                if not package:
                    continue
                install_package(env, package)

    with open("requirements/model-requirements.txt", "r") as file:
        for line in file:
            MODEL_URL = line.strip()
            if not MODEL_URL:
                continue

            DEST_PATH = "src/Include/models/" + MODEL_URL.split("/")[-1]
            os.makedirs("src/Includemodels", exist_ok=True)
            download_model(MODEL_URL, DEST_PATH)
