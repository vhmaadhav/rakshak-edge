"""rbench: benchmark harness for Rakshak Edge on the Raspberry Pi 5.

Pick a registered model and dataset, run them under the thermal guard, and get a comparable
JSON result with system telemetry. See deploy/pi/bench/README.md.
"""

__version__ = "0.1.0"
SCHEMA = "rbench/1"
REGISTRY_SCHEMA = "rbench-registry/1"

# Exit codes (documented in README.md; stable for agents).
EXIT_OK = 0
EXIT_USAGE = 2  # bad usage, incompatible model/dataset, disabled entry
EXIT_MISSING = 3  # missing artifact, sha mismatch, or missing dependency
EXIT_THERMAL = 4  # the thermal guard killed the run
EXIT_RUNTIME = 5  # runtime error, or the bench lock is held by another run
