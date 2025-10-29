import os
import sys
import subprocess
import time

def install_package(env: list[str], package: list[str], retries: int = 3):
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
                print(f"{package} Install failed after multiple attempts.\n")

    print(f"Failed to install {package} after {retries} attempts.\n")
    return False

def extract_packages(requirements_dir: str) -> list[tuple[list, list]]:
    if not os.path.exists(requirements_dir):
        raise FileNotFoundError(f'requirements file at "{requirements_dir}" not found')
    if os.path.isdir(requirements_dir):
        raise FileNotFoundError(f'requirements path "{requirements_dir}" cannot be a folder')

    with open(requirements_dir, "r") as file:
        env_packages = []

        for line in file:
            env_package = line.strip().split('@')
            if not env_package:
                continue

            if len(env_package) == 1:
                env, package = [], env_package[0].split(';')
            else:
                env, package = env_package[0].split(';'), env_package[1].split(';')
            
            if not package:
                continue
            
            env_packages.append((env, package))

        return env_packages

if __name__ == "__main__":
    requirements_dir = os.path.join("dev", "requirements", "python_requirements.txt")
    env_packages = extract_packages(requirements_dir)

    for env, package in env_packages:
        install_package(env, package)