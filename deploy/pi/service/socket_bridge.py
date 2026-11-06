#!/usr/bin/env python3
"""Local TCP-to-Unix-socket relay for the edge service (stdlib only, runs outside its namespace).

The edge service runs in ``unshare -rn`` and has no network. Its HTTP API is also served on a
Unix socket in the run directory. This relay listens on 127.0.0.1 (default port 8770) and copies
bytes to that socket, so a browser on the laptop reaches the page through an SSH tunnel:

    ssh -i ~/.ssh/pi5 -N -L 8770:127.0.0.1:8770 hermoine@pi5.local
    open http://localhost:8770

It only relays bytes between loopback and the socket. It makes no outbound connection.
"""

from __future__ import annotations

import argparse
import socket
import threading


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for s, how in ((dst, socket.SHUT_WR), (src, socket.SHUT_RD)):
            try:
                s.shutdown(how)
            except OSError:
                pass


def _handle(client: socket.socket, unix_path: str) -> None:
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(unix_path)
    except OSError:
        client.close()
        upstream.close()
        return
    back = threading.Thread(target=_pipe, args=(upstream, client), daemon=True)
    back.start()
    _pipe(client, upstream)
    back.join()
    client.close()
    upstream.close()


def serve(host: str, port: int, unix_path: str) -> None:
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit("socket_bridge: listen on loopback only")
    listener = socket.create_server((host, port))
    try:
        while True:
            client, _ = listener.accept()
            threading.Thread(target=_handle, args=(client, unix_path), daemon=True).start()
    finally:
        listener.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--listen", default="127.0.0.1:8770")
    p.add_argument("--unix", required=True, help="the edge service's Unix socket path")
    args = p.parse_args(argv)
    host, _, port = args.listen.rpartition(":")
    serve(host, int(port), args.unix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
