"""Helpers for test/lit.cfg.py that lit's worker processes must import.

lit sends each test's configuration to its workers by pickling it, and
pickles a function or an object by its module and name, so what the
configuration holds has to be in a module the workers can import, not in
lit.cfg.py itself.
"""

import errno
import fcntl
import os
import time

import lit.formats

from gpu_info import gpu_unit_mask, visible_gpus  # noqa: F401, for lit.cfg.py


def gpu_parallelism_group(test):
    """The parallelism group of the GPU tests, the -gpu.test files."""
    if test.getFilePath().endswith("-gpu.test"):
        return "merkurius-gpu"
    return None


class GpuShTest(lit.formats.ShTest):
    """lit's shell tests, which run each GPU test, a -gpu.test file, on one
    of the GPUs, by ROCR_VISIBLE_DEVICES: `per_gpu` GPU tests at most on
    each, taken by locking one of `slots` lock files in `lock_dir`, slot i
    being the GPU gpu_ids[i % len(gpu_ids)]. lit's parallelism group runs no
    more GPU tests at once than there are slots, so a free one is always
    there, but for another lit run of the same build's tests."""

    def __init__(self, gpu_ids, per_gpu, lock_dir):
        super().__init__()
        self.gpu_ids = list(gpu_ids)
        self.slots = len(self.gpu_ids) * per_gpu
        self.lock_dir = lock_dir

    def _take_slot(self):
        os.makedirs(self.lock_dir, exist_ok=True)
        while True:
            for slot in range(self.slots):
                path = os.path.join(self.lock_dir, "gpu-slot-%d" % slot)
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd, self.gpu_ids[slot % len(self.gpu_ids)]
                except OSError as error:
                    os.close(fd)
                    if error.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
            time.sleep(0.1)

    def execute(self, test, litConfig):
        if not self.gpu_ids or not test.getFilePath().endswith("-gpu.test"):
            return super().execute(test, litConfig)
        fd, gpu = self._take_slot()
        # The worker runs one test at a time, on its own copy of the
        # configuration.
        environment = test.config.environment
        before = environment.get("ROCR_VISIBLE_DEVICES")
        environment["ROCR_VISIBLE_DEVICES"] = gpu
        try:
            return super().execute(test, litConfig)
        finally:
            if before is None:
                del environment["ROCR_VISIBLE_DEVICES"]
            else:
                environment["ROCR_VISIBLE_DEVICES"] = before
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
