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

from gpu_info import gpu_unit_mask, one_unit_mask, visible_gpus  # noqa: F401, for lit.cfg.py


def gpus_of(test):
    """How many GPUs a test runs on: 1 for a -gpu.test file, 2 for a
    -two-gpus.test file, else 0."""
    path = test.getFilePath()
    if path.endswith("-gpu.test"):
        return 1
    if path.endswith("-two-gpus.test"):
        return 2
    return 0


def gpu_parallelism_group(test):
    """The parallelism group of the GPU tests, the -gpu.test and
    -two-gpus.test files."""
    if gpus_of(test):
        return "merkurius-gpu"
    return None


class GpuShTest(lit.formats.ShTest):
    """lit's shell tests, which run each GPU test, a -gpu.test file, on one
    of the GPUs, by ROCR_VISIBLE_DEVICES: `per_gpu` GPU tests at most on
    each, taken by locking one of `slots` lock files in `lock_dir`, slot i
    being the GPU gpu_ids[i % len(gpu_ids)]. lit's parallelism group runs no
    more GPU tests at once than there are slots, so a free one is always
    there, but for another lit run of the same build's tests. A test of two
    GPUs, a -two-gpus.test file, takes two slots of different GPUs, and sees
    both, in the order of their slots; it waits while it cannot have both,
    holding neither, so that two such tests never each hold one."""

    def __init__(self, gpu_ids, per_gpu, lock_dir):
        super().__init__()
        self.gpu_ids = list(gpu_ids)
        self.slots = len(self.gpu_ids) * per_gpu
        self.lock_dir = lock_dir

    def _take_slots(self, count):
        """Locks `count` slots of different GPUs: their files and GPUs."""
        os.makedirs(self.lock_dir, exist_ok=True)
        while True:
            taken = []
            for slot in range(self.slots):
                gpu = self.gpu_ids[slot % len(self.gpu_ids)]
                if any(gpu == other for _, other in taken):
                    continue
                path = os.path.join(self.lock_dir, "gpu-slot-%d" % slot)
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    taken.append((fd, gpu))
                except OSError as error:
                    os.close(fd)
                    if error.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                if len(taken) == count:
                    return taken
            self._release(taken)
            time.sleep(0.1)

    @staticmethod
    def _release(taken):
        for fd, _ in taken:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def execute(self, test, litConfig):
        count = gpus_of(test)
        if not self.gpu_ids or count == 0:
            return super().execute(test, litConfig)
        # A machine with fewer GPUs runs no such test: its REQUIRES says so.
        count = min(count, len(self.gpu_ids))
        taken = self._take_slots(count)
        # The worker runs one test at a time, on its own copy of the
        # configuration.
        environment = test.config.environment
        before = environment.get("ROCR_VISIBLE_DEVICES")
        environment["ROCR_VISIBLE_DEVICES"] = ",".join(gpu for _, gpu in taken)
        try:
            return super().execute(test, litConfig)
        finally:
            if before is None:
                del environment["ROCR_VISIBLE_DEVICES"]
            else:
                environment["ROCR_VISIBLE_DEVICES"] = before
            self._release(taken)
