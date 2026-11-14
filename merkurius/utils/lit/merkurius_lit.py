"""Helpers for test/lit.cfg.py that lit's worker processes must import.

lit sends each test's configuration to its workers by pickling it, and
pickles a function or an object by its module and name, so what the
configuration holds has to be in a module the workers can import, not in
lit.cfg.py itself.
"""

import errno
import fcntl
import glob
import os
import time

import lit.formats


def gpu_parallelism_group(test):
    """The parallelism group of the GPU tests, the -gpu.test files."""
    if test.getFilePath().endswith("-gpu.test"):
        return "merkurius-gpu"
    return None


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _properties(node):
    result = {}
    for line in _read(os.path.join(node, "properties")).splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = fields[1]
    return result


def visible_gpus():
    """This machine's GPUs that the ROCm runtime shows the tests, in its
    order: the KFD topology's nodes that have a GPU, those that
    ROCR_VISIBLE_DEVICES names if it is set, as utils/gpu/with-gpus.py sets
    it to numbers. Each is a dict: "id", the number that
    ROCR_VISIBLE_DEVICES takes, "units", its compute units, and "display",
    whether it drives a display."""
    nodes = glob.glob("/sys/class/kfd/kfd/topology/nodes/*")
    nodes.sort(key=lambda node: int(os.path.basename(node)))
    gpus = []
    for node in nodes:
        props = _properties(node)
        if props.get("gfx_target_version", "0") == "0":
            continue
        simds = int(props.get("simd_count", "0"))
        per_unit = int(props.get("simd_per_cu", "1")) or 1
        gpus.append({
            "id": str(len(gpus)),
            "units": simds // per_unit,
            "display": _drives_display(props.get("drm_render_minor")),
        })
    chosen = os.environ.get("ROCR_VISIBLE_DEVICES")
    if chosen is not None:
        by_id = {gpu["id"]: gpu for gpu in gpus}
        gpus = [by_id[i] for i in chosen.split(",") if i in by_id]
    return gpus


def _drives_display(render_minor):
    """Whether the GPU of the DRM render node renderD<render_minor> has a
    connected display: a connector of its own card, not of another device's,
    such as a server's management controller's VGA."""
    if render_minor is None:
        return False
    device = "/sys/class/drm/renderD%s/device/drm" % render_minor
    for card in glob.glob(os.path.join(device, "card*")):
        name = os.path.basename(card)
        for status in glob.glob("/sys/class/drm/%s-*/status" % name):
            if _read(status) == "connected":
                return True
    return False


def gpu_unit_mask(gpus):
    """The compute units that GPU processes take in the tests, as
    LIBOMPTARGET_AMDGPU_CU_MASK gives them (runtime/compute-units.h): half
    of the units if a GPU drives a display, so that the display keeps the
    rest, else None, for all of them. One mask for every GPU, from the
    smallest."""
    if not any(gpu["display"] for gpu in gpus):
        return None
    units = min(gpu["units"] for gpu in gpus)
    return "0-%d" % (max(units // 2, 2) - 1)


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
