#!/usr/bin/env python3
"""Expose a Windows-loopback port to WSL without modifying HTTP traffic."""

import argparse
import socket
import threading


def copy(source, destination):
    try:
        while data := source.recv(65536):
            destination.sendall(data)
    except OSError:
        pass
    for stream in (source, destination):
        try: stream.shutdown(socket.SHUT_RDWR)
        except OSError: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.listen))
    server.listen(16)
    print(f"ready {args.listen}->{args.target}", flush=True)
    while True:
        client, _ = server.accept()
        upstream = socket.create_connection(("127.0.0.1", args.target), timeout=5)
        upstream.settimeout(None)
        threading.Thread(target=copy, args=(client, upstream), daemon=True).start()
        threading.Thread(target=copy, args=(upstream, client), daemon=True).start()


if __name__ == "__main__":
    main()
