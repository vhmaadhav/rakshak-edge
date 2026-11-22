#!/usr/bin/env bash
# OS-level offline check.
#
# Runs a pytest selection (default: the SMS/URL/flow smoke below) inside a fresh network namespace
# that has no interfaces except loopback. The namespace is the stronger check: it catches native
# code (ONNX Runtime, BLAS, C libraries) that opens sockets without going through Python. The
# Python-level socket patch in tests/test_no_egress.py is only a regression layer.
#
# Usage:
#   scripts/offline_check.sh                          # default smoke selection
#   scripts/offline_check.sh tests/test_url_cache_bounds.py -k url
#
# Exit status: 0 only if the negative network control fails as expected and every test passes.
# Needs util-linux unshare and unprivileged user namespaces (no root). If unavailable the script
# stops with exit 3. Set OFFLINE_CHECK_ALLOW_NO_NAMESPACE=1 to run the selection without isolation;
# the run is then labelled NOT OFFLINE-VERIFIED.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_SELECTION=(
    tests/test_no_egress.py
    tests/test_url_cache_bounds.py
    tests/test_frozen_url_contract.py
    tests/test_edge_url_model.py
    tests/test_edge_service.py
)

if [[ "$#" -gt 0 ]]; then
    SELECTION=("$@")
else
    SELECTION=("${DEFAULT_SELECTION[@]}")
fi

PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "offline_check: no interpreter at $PY (set PYTHON or create .venv)" >&2
    exit 2
fi

inner() {
    # Inside the namespace. Loopback is down in a fresh namespace; bring it up so localhost
    # tests (the edge HTTP surface) still work. Everything else stays unreachable.
    if command -v ip >/dev/null 2>&1; then
        ip link set lo up 2>/dev/null || true
    fi

    echo "offline_check: negative control (a non-loopback connect must fail)"
    if "$PY" -c "
import socket, sys
try:
    socket.create_connection(('1.1.1.1', 53), timeout=3)
except OSError as exc:
    print('  refused as expected:', exc)
    sys.exit(0)
print('  FAIL: outbound connect succeeded; namespace is not isolated')
sys.exit(1)
"; then
        :
    else
        echo "offline_check: FAILED (network namespace is not isolated)" >&2
        exit 1
    fi

    echo "offline_check: pytest ${SELECTION[*]}"
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -q -rs "${SELECTION[@]}"
    echo "offline_check: OK (all selected tests passed inside the network namespace)"
}

if [[ "${OFFLINE_CHECK_INNER:-0}" == "1" ]]; then
    inner
    exit 0
fi

if command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
    OFFLINE_CHECK_INNER=1 unshare -rn -- "$0" "$@"
    exit $?
fi

if [[ "${OFFLINE_CHECK_ALLOW_NO_NAMESPACE:-0}" == "1" ]]; then
    echo "offline_check: WARNING unshare -rn unavailable; running WITHOUT network isolation." >&2
    echo "offline_check: NOT OFFLINE-VERIFIED. Only the Python-level regression layer ran." >&2
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -q -rs "${SELECTION[@]}"
    exit $?
fi

echo "offline_check: cannot create a network namespace (unshare -rn failed or is missing)." >&2
echo "offline_check: needs util-linux 'unshare' with unprivileged user namespaces enabled." >&2
echo "offline_check: set OFFLINE_CHECK_ALLOW_NO_NAMESPACE=1 to run without isolation (not verified)." >&2
exit 3
