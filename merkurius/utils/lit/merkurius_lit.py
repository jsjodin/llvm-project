"""Helpers for test/lit.cfg.py that lit's worker processes must import.

lit sends each test's configuration to its workers by pickling it, and
pickles a function by its module and name, so a function that the
configuration holds has to be in a module the workers can import, not in
lit.cfg.py itself.
"""


def gpu_parallelism_group(test):
    """The parallelism group of the GPU tests, the -gpu.test files."""
    if test.getFilePath().endswith("-gpu.test"):
        return "merkurius-gpu"
    return None
