import json
import socket

MAX_MESSAGE = 2 * 1024 * 1024


def receive(connection):
    data = bytearray()
    while b"\n" not in data:
        block = connection.recv(min(65536, MAX_MESSAGE-len(data)))
        if not block:
            raise ConnectionError("Incomplete worker message")
        data.extend(block)
        if len(data) >= MAX_MESSAGE:
            raise ValueError("Worker message too large")
    return json.loads(data.split(b"\n", 1)[0])


def send(connection, value):
    encoded = json.dumps(value, ensure_ascii=False).encode()+b"\n"
    if len(encoded) >= MAX_MESSAGE:
        raise ValueError("Worker response too large")
    connection.sendall(encoded)


def request(path, payload, timeout):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(path)
        send(client, payload)
        return receive(client)
