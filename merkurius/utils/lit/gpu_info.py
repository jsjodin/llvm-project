"""This machine's GPUs as the tests and benchmarks use them, from sysfs
alone: which the ROCm runtime shows, their compute units, and the units that
GPU processes take. test/lit.cfg.py imports it (through merkurius_lit.py);
a script runs it:

  python3 -I utils/lit/gpu_info.py unit-mask   # the mask, or nothing

which prints the units for LIBOMPTARGET_AMDGPU_CU_MASK, or nothing for all
of them.
"""

import glob
import os
import sys


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


if __name__ == "__main__":
    if sys.argv[1:] == ["unit-mask"]:
        gpus = visible_gpus()
        mask = gpu_unit_mask(gpus) if gpus else None
        if mask:
            print(mask)
    else:
        sys.exit("usage: gpu_info.py unit-mask")
