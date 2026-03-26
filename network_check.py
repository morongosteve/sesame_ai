import socket


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


def test_port(port: int = 8188) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("localhost", port))
    if result == 0:
        print(f"❌ Port {port} already in use")
    else:
        print(f"✅ Port {port} available")


print("Local IP:", get_local_ip())
test_port()
