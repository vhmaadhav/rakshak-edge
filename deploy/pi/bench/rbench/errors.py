"""Exception types shared by the worker and the CLI."""


class BenchError(RuntimeError):
    """A runtime failure inside the benchmark (exit code 5)."""


class MissingDependency(BenchError):
    """A Python package the runner needs is not importable (exit code 3)."""


class ArtifactError(BenchError):
    """A model or dataset file is missing or fails its sha256 pin (exit code 3)."""
