import os
import requests
import sys
import time

def download_model(url: str, dest_path: str, retries: int = 3, timeout: int = 30):
    if os.path.exists(dest_path):
        print(f"Model already exists at {dest_path}\n")
        answer = input("Would you like to reinstall? (Y/N): ").lower()
        if answer == 'n':
            return
        elif answer != 'y':
            print('Received a different answer then (Y/N), skipping reinstalltion')
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
    with open("requirements/model-requirements.txt", "r") as file:
        for line in file:
            MODEL_URL = line.strip()
            if not MODEL_URL:
                continue

            DEST_PATH = "models/" + MODEL_URL.split("/")[-1]
            os.makedirs("models/", exist_ok=True)
            download_model(MODEL_URL, DEST_PATH)
