#!/usr/bin/env python3
"""Runs a command on GPUs of this machine that no one else is using.

On a machine that others share, every test, benchmark and demo run goes
through this script, so that it uses only the GPUs chosen for it:

  python3 -I utils/gpu/with-gpus.py                  # list the GPUs
  python3 -I utils/gpu/with-gpus.py --count 2 -- llvm-lit ...
  python3 -I utils/gpu/with-gpus.py --gpus 2,3 -- ./prog

It sets ROCR_VISIBLE_DEVICES to the chosen GPUs, which the ROCm runtime
honours below both OpenMP's offloading and liboffload, so target regions
and GPU processes alike see only those, as GPUs 0, 1 and so on.

A GPU is free when no other process had memory on it, by the KFD driver's
files of each process in sysfs, and when it was busy at most --max-busy
percent of the time, both over a second of samples. It reads sysfs alone, with no SMI tool.
A GPU that drives a display is busy for a few percent with no process on
it, which the default allows.

The command holds a lock on each of its GPUs in /tmp/merkurius-gpus-<uid>
until it ends, so that two of one user's runs never take the same GPU.
Other users' programs do not take the lock; they are seen by their memory
on the GPU once they run.

The GPUs are numbered as the ROCm runtime numbers them, in the order of the
KFD topology's nodes, and ROCR_VISIBLE_DEVICES takes those numbers.
"""

import argparse
import fcntl
import os
import pwd
import sys
import time


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def properties(node_dir):
    result = {}
    for line in (read(os.path.join(node_dir, "properties"), "") or
                 "").splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = fields[1]
    return result


def gfx_name(version):
    """gfx_target_version as --offload-arch names it: 90010 is gfx90a."""
    major, minor, stepping = version // 10000, version // 100 % 100, \
        version % 100
    return "gfx%d%x%x" % (major, minor, stepping)


class Gpu:
    def __init__(self, index, node_dir, props, root):
        self.index = index
        self.gpu_id = read(os.path.join(node_dir, "gpu_id"), "0")
        self.name = gfx_name(int(props.get("gfx_target_version", "0")))
        device = os.path.join(root, "sys/class/drm",
                              "renderD" + props.get("drm_render_minor", ""),
                              "device")
        self.busy_file = os.path.join(device, "gpu_busy_percent")
        self.vram_used = int(read(os.path.join(device, "mem_info_vram_used"),
                                  "0"))
        self.vram_total = int(read(os.path.join(device,
                                                "mem_info_vram_total"), "0"))
        self.busy = 0.0
        self.others = []  # (pid, user, command, bytes)


def find_gpus(root):
    """The machine's GPUs, numbered as the ROCm runtime numbers them."""
    nodes = os.path.join(root, "sys/class/kfd/kfd/topology/nodes")
    try:
        names = sorted(os.listdir(nodes), key=int)
    except OSError:
        return []
    gpus = []
    for name in names:
        node_dir = os.path.join(nodes, name)
        props = properties(node_dir)
        if int(props.get("simd_count", "0")) > 0:
            gpus.append(Gpu(len(gpus), node_dir, props, root))
    return gpus


def find_users(gpus, root):
    """Adds each GPU's other processes, with the memory they hold on it."""
    procs = os.path.join(root, "sys/class/kfd/kfd/proc")
    try:
        pids = os.listdir(procs)
    except OSError:
        return
    for pid in pids:
        if pid == str(os.getpid()):
            continue
        for gpu in gpus:
            used = read(os.path.join(procs, pid, "vram_" + gpu.gpu_id))
            if used is None or int(used) == 0 or \
                    any(other[0] == pid for other in gpu.others):
                continue
            command = read(os.path.join(root, "proc", pid, "comm"), "?")
            try:
                uid = os.stat(os.path.join(root, "proc", pid)).st_uid
                user = pwd.getpwuid(uid).pw_name
            except (OSError, KeyError):
                user = "?"
            gpu.others.append((pid, user, command, int(used)))


def sample(gpus, root, seconds, samples):
    """How busy each GPU was, and who used it, over `seconds`."""
    totals = [0.0] * len(gpus)
    for i in range(samples):
        find_users(gpus, root)
        for j, gpu in enumerate(gpus):
            totals[j] += float(read(gpu.busy_file, "0") or 0)
        if i + 1 < samples:
            time.sleep(seconds / samples)
    for j, gpu in enumerate(gpus):
        gpu.busy = totals[j] / samples


def mib(n):
    return n // (1024 * 1024)


def show(gpus, max_busy, out):
    print("GPU  arch      busy  memory used        other processes", file=out)
    for gpu in gpus:
        state = "free" if is_free(gpu, max_busy) else "busy"
        others = ", ".join("%s %s (pid %s, %d MiB)" %
                           (user, command, pid, mib(used))
                           for pid, user, command, used in gpu.others)
        print("%-4d %-8s %4.0f%%  %6d/%6d MiB  %s%s" %
              (gpu.index, gpu.name, gpu.busy, mib(gpu.vram_used),
               mib(gpu.vram_total), state, ": " + others if others else ""),
              file=out)


def is_free(gpu, max_busy):
    return not gpu.others and gpu.busy <= max_busy


def lock(gpu, directory):
    """A lock on the GPU for this user's runs, or None if another holds it."""
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd = os.open(os.path.join(directory, "%d.lock" % gpu.index),
                 os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.set_inheritable(fd, True)
    return fd


def main():
    parser = argparse.ArgumentParser(
        description="Run a command on free GPUs of this machine.")
    choose = parser.add_mutually_exclusive_group()
    choose.add_argument("--count", type=int,
                        help="how many free GPUs the command gets")
    choose.add_argument("--gpus",
                        help="the GPUs the command gets, as 2,3")
    parser.add_argument("--max-busy", type=float, default=10.0,
                        help="the most a free GPU may be busy, in percent "
                        "(default 10)")
    parser.add_argument("--force", action="store_true",
                        help="with --gpus, use them even if they are busy")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the choice, and run nothing")
    parser.add_argument("--root", default="/", help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=10,
                        help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- and the command to run")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else \
        args.command

    if "ROCR_VISIBLE_DEVICES" in os.environ:
        sys.exit("with-gpus: ROCR_VISIBLE_DEVICES is already set, to " +
                 os.environ["ROCR_VISIBLE_DEVICES"] +
                 "; unset it, as this script numbers every GPU")
    gpus = find_gpus(args.root)
    if not gpus:
        sys.exit("with-gpus: no GPU found in the KFD topology")
    sample(gpus, args.root, 1.0, max(args.samples, 1))

    if args.count is None and args.gpus is None:
        show(gpus, args.max_busy, sys.stdout)
        if command:
            sys.exit("with-gpus: give --count or --gpus to run a command")
        return

    if not command and not args.dry_run:
        sys.exit("with-gpus: no command to run, after --")

    directory = os.path.join(args.root, "tmp",
                             "merkurius-gpus-%d" % os.getuid())
    chosen, locks = [], []
    if args.gpus is not None:
        for text in args.gpus.split(","):
            if not text.isdigit() or int(text) >= len(gpus):
                sys.exit("with-gpus: there is no GPU " + text +
                         "; the machine has %d" % len(gpus))
            gpu = gpus[int(text)]
            if gpu in chosen:
                sys.exit("with-gpus: GPU %d is named twice" % gpu.index)
            if not args.force and not is_free(gpu, args.max_busy):
                show(gpus, args.max_busy, sys.stderr)
                sys.exit("with-gpus: GPU %d is in use; --force uses it "
                         "anyway" % gpu.index)
            fd = lock(gpu, directory)
            if fd is None:
                sys.exit("with-gpus: another of your runs holds GPU %d" %
                         gpu.index)
            chosen.append(gpu)
            locks.append(fd)
    else:
        if args.count < 1:
            sys.exit("with-gpus: --count must be at least 1")
        for gpu in gpus:
            if len(chosen) == args.count:
                break
            if not is_free(gpu, args.max_busy):
                continue
            fd = lock(gpu, directory)
            if fd is not None:
                chosen.append(gpu)
                locks.append(fd)
        if len(chosen) < args.count:
            show(gpus, args.max_busy, sys.stderr)
            sys.exit("with-gpus: only %d of the %d GPUs asked for are free" %
                     (len(chosen), args.count))

    visible = ",".join(str(gpu.index) for gpu in chosen)
    print("with-gpus: ROCR_VISIBLE_DEVICES=%s (%s)" %
          (visible, ", ".join(gpu.name for gpu in chosen)), file=sys.stderr)
    if args.dry_run:
        return
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=visible)
    try:
        os.execvpe(command[0], command, env)
    except OSError as e:
        sys.exit("with-gpus: cannot run %s: %s" % (command[0], e.strerror))


if __name__ == "__main__":
    main()
