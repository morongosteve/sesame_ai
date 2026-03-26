import os
import platform
import socket
import subprocess
import sys


def run_cmd(cmd: str) -> str:
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return result.decode().strip()
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.output.decode()}"


def check_os() -> None:
    print("=== OS CHECK ===")
    print(platform.system(), platform.release())


def check_python() -> None:
    print("=== PYTHON CHECK ===")
    print(sys.version)
    if sys.version_info < (3, 10):
        print("❌ Python too old. Install 3.10+")
        sys.exit(1)


def check_gpu() -> bool:
    print("=== GPU CHECK ===")
    result = run_cmd("nvidia-smi")
    if "ERROR" in result:
        print("❌ NVIDIA GPU not detected or drivers missing")
        return False
    print(result)
    return True


def check_cuda() -> bool:
    print("=== CUDA CHECK ===")
    result = run_cmd("nvcc --version")
    if "ERROR" in result:
        print("⚠️ CUDA not installed or not in PATH")
        return False
    print(result)
    return True


def check_ram() -> None:
    print("=== RAM CHECK ===")
    try:
        import psutil

        ram = psutil.virtual_memory().total / (1024**3)
        print(f"{ram:.2f} GB RAM")
        if ram < 16:
            print("⚠️ Low RAM (<16GB)")
    except Exception:
        print("⚠️ psutil not installed")


def check_disk() -> None:
    print("=== DISK CHECK ===")
    stat = os.statvfs("/")
    free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
    print(f"Free disk: {free_gb:.2f} GB")
    if free_gb < 20:
        print("❌ Not enough disk space")
        sys.exit(1)


def check_port(port: int = 8188) -> None:
    print(f"=== PORT CHECK ({port}) ===")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("localhost", port))
        if result == 0:
            print(f"⚠️ Port {port} already in use")
        else:
            print("✅ Port available")


def check_network() -> None:
    print("=== NETWORK CHECK ===")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        print(f"Local IP: {ip}")
    except Exception:
        print("⚠️ Could not determine IP")


def main() -> None:
    check_os()
    check_python()
    gpu_ok = check_gpu()
    check_cuda()
    check_ram()
    check_disk()
    check_port()
    check_network()

    if not gpu_ok:
        print("🚫 STOP: Fix GPU before continuing")
        sys.exit(1)


if __name__ == "__main__":
    main()
