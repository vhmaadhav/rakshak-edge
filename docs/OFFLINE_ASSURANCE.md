# Offline assurance

Rakshak runs its models on the device (laptop, Raspberry Pi, Android app). This page says what the
offline checks in the repository cover, what they do not cover, and how to run them.

## The two checks

**1. OS-level: `scripts/offline_check.sh`** (the stronger check)

- Runs a pytest selection inside a fresh network namespace created with `unshare -rn`. The
  namespace has no interfaces except loopback. No root is needed.
- Before the selection runs, a negative control tries a non-loopback connect. The run stops with
  a failure if that connect succeeds.
- Default selection: `tests/test_no_egress.py`, `tests/test_url_cache_bounds.py`,
  `tests/test_frozen_url_contract.py`, `tests/test_edge_url_model.py`, `tests/test_edge_service.py`.
  Pass other paths or pytest flags as arguments to run a different selection.
- Exits non-zero if the negative control fails or any selected test fails.
- If `unshare -rn` is unavailable the script exits with status 3 and says why. The
  `OFFLINE_CHECK_ALLOW_NO_NAMESPACE=1` variable runs the selection without isolation and labels the
  result NOT OFFLINE-VERIFIED.

**2. Python-level regression: `tests/test_no_egress.py`**

- Patches `socket.socket.connect`, `connect_ex`, `socket.create_connection`, `getaddrinfo` and
  `gethostbyname` to raise, then loads the shipped models through the code paths the other tests
  use and runs one inference each.
- Skips cleanly when a model file is not in the checkout. Model files are not stored in the
  repository.
- This layer only catches Python code that uses the `socket` module. It is a regression guard, not
  proof of isolation.

## What the checks cover

- Python code in `src/rakshak` and `deploy/pi` that is exercised by the selected tests.
- Model loading and one inference for the SMS and URL models, when the model files are present.
- The edge HTTP surface and URL path in the selected tests (loopback only).

## What the checks do not cover

- **Native libraries.** ONNX Runtime, BLAS and other C/C++ code can open sockets without going
  through Python. The namespace check covers this for the processes it runs, but only for the code
  paths the selection exercises.
- **Subprocesses started outside the selection.** A child process inherits the namespace, but a
  path not covered by the selection is not tested.
- **Other machines.** A pass in this namespace says nothing about the Raspberry Pi image, its
  packages or its system services.
- **Final Android APK permissions.** These need a device check on the built APK.

## Android manifest

`android/app/src/main/AndroidManifest.xml` declares, for the merged app:

- `android.permission.INTERNET`: removed (`tools:node="remove"`).
- `android.permission.ACCESS_NETWORK_STATE`: removed (`tools:node="remove"`).
- The `ai.onnxruntime.TelemetryInitializer` content provider: removed (`tools:node="remove"`).

The manifest's own comment says the app receives messages through paste or the system share
sheet (`ACTION_SEND`, `text/plain`) and that ONNX Runtime runs the bundled model locally.

These `remove` markers only take effect in the merged manifest that the build produces. Check the
merged manifest of the built APK, and the permissions the installed app reports on a device, before
relying on them. This check is outside what the repository tests can show.

## How to run

```bash
# Stronger check: default smoke selection inside a network namespace
scripts/offline_check.sh

# A different selection
scripts/offline_check.sh tests/test_url_cache_bounds.py -k eviction

# Regression layer alone, on the normal host network
.venv/bin/python -m pytest -q tests/test_no_egress.py
```

Set `PYTHONPATH=src` when running pytest from a git worktree, so the worktree's `src` is imported
instead of an installed copy.
