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
import glob
import itertools
import os
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


def upstream_paths(unix_path: str) -> list[str]:
    """The service sockets to relay to: ``<unix_path>.<n>`` (one per verdict worker, sorted) when
    they exist, and ``<unix_path>`` itself when it exists or when nothing else does. A stale
    ``<unix_path>.<n>`` left by a killed multi-worker service therefore cannot hide the socket of
    a later single-worker service: the connect loop skips a path that does not accept."""
    found = sorted(glob.glob(unix_path + ".*"), key=lambda p: (len(p), p))
    if os.path.exists(unix_path) or not found:
        found.append(unix_path)
    return found


_RR = itertools.count()
_RR_LOCK = threading.Lock()


def _connect(unix_path: str) -> socket.socket | None:
    """A connection to the next socket in round-robin order; the others are tried in turn when a
    worker's socket does not accept (for example a worker that is restarting)."""
    paths = upstream_paths(unix_path)
    with _RR_LOCK:
        start = next(_RR)
    for k in range(len(paths)):
        path = paths[(start + k) % len(paths)]
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(path)
            return upstream
        except OSError:
            upstream.close()
    return None


def _handle(client: socket.socket, unix_path: str) -> None:
    # Small responses must not wait for Nagle: the relay forwards each chunk as it arrives.
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    upstream = _connect(unix_path)
    if upstream is None:
        client.close()
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
    # Keep-alive clients reuse one connection per session, so the relay thread is per connection.
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
