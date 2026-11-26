#!/usr/bin/env python3
"""Runs a program's nodes and kills them with SIGKILL at random times.

Usage: kill-nodes.py [--seed N] [--mtbf SECONDS] [--count N]
                     [--spare N]... [--port-from N]
                     [--log FILE] [--replay FILE] -- COMMAND [COMMAND...]

The fault injector's harness half (docs/INPROGRESS.md, Failure, step 3):
the runtime's MERK_FAULTS fails processes and silences connections inside
a node, and this ends whole nodes, as a crash or a lost machine would.

Each COMMAND is a node, split as a shell would split it, and run with this
script's standard streams. At exponentially distributed times, from the
seed, with the mean time between failures in seconds, it kills a random
node that is still running, until it has killed --count of them, 1 by
default. The schedule has a line for each kill, the time in milliseconds
since the nodes started and the node's number among the commands, from 0,
after a line with the seed. --replay kills the same nodes at the same
times, and writes what it did to --log too, without the seed's line.

--spare N keeps node N from being killed, as a program's coordinator,
which may be given more than once. --port-from N starts node N first and
waits for it to print "listening PORT", as a node that listens on a free
port does; the other commands' {port} is that port, and the clock of the
kills starts once they all run. Node N's output still goes to this
script's standard output.

Once every node has ended, it prints how each ended, and exits with 0 if
every node it did not kill exited with 0, else 1.
"""

import argparse
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

POLL = 0.01


def arguments():
    parser = argparse.ArgumentParser(
        description="Run nodes and kill them with SIGKILL at random times.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mtbf", type=float,
                        help="the mean time between failures, in seconds")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--spare", type=int, action="append", default=[],
                        help="a node that is never killed")
    parser.add_argument("--port-from", type=int,
                        help="the node whose \"listening PORT\" is {port}")
    parser.add_argument("--log", help="where the schedule is written")
    parser.add_argument("--replay", help="a schedule to repeat")
    parser.add_argument("commands", nargs="+", metavar="COMMAND")
    options = parser.parse_args()
    if options.replay:
        if options.seed is not None or options.mtbf is not None:
            parser.error("--replay takes no --seed or --mtbf")
    elif options.mtbf is None or options.mtbf <= 0:
        parser.error("--mtbf SECONDS is missing")
    return options


def read_schedule(path):
    kills = []
    with open(path) as file:
        for line in file:
            words = line.split()
            if not words or words[0].startswith("#"):
                continue
            if len(words) != 3 or words[1] != "kill":
                sys.exit(f"kill-nodes.py: not a line of a schedule: {line!r}")
            kills.append((int(words[0]) / 1000, int(words[2])))
    return kills


def wait_until(nodes, when):
    """Sleeps until `when`, or until every node has ended: whether it is
    `when`."""
    while time.monotonic() < when:
        if all(node.poll() is not None for node in nodes):
            return False
        time.sleep(min(POLL, max(0, when - time.monotonic())))
    return True


def start(commands, port_from):
    """The nodes, started, and the thread that forwards the output of the
    node whose port the others' {port} is, if there is one."""
    if port_from is None:
        return [subprocess.Popen(shlex.split(c)) for c in commands], None
    if not 0 <= port_from < len(commands):
        sys.exit(f"kill-nodes.py: --port-from {port_from}: there is no such "
                 "node")
    listener = subprocess.Popen(shlex.split(commands[port_from]),
                                stdout=subprocess.PIPE, text=True)
    port = None
    for line in listener.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = re.match(r"listening (\d+)$", line.strip())
        if m:
            port = m.group(1)
            break
    if port is None:
        listener.wait()
        sys.exit(f"kill-nodes.py: node {port_from} ended without printing "
                 "\"listening PORT\"")

    def copy():
        for line in listener.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    forward = threading.Thread(target=copy, daemon=True)
    forward.start()
    nodes = [listener if i == port_from
             else subprocess.Popen(shlex.split(c.replace("{port}", port)))
             for i, c in enumerate(commands)]
    return nodes, forward


def main():
    options = arguments()
    seed = options.seed
    if seed is None and not options.replay:
        seed = random.SystemRandom().randrange(2**63)
    log = open(options.log, "w") if options.log else None
    if log and not options.replay:
        print(f"# seed={seed}", file=log, flush=True)
    nodes, forward = start(options.commands, options.port_from)
    started = time.monotonic()
    killed = set()

    def kill(at, index):
        nodes[index].send_signal(signal.SIGKILL)
        killed.add(index)
        line = f"{round((at - started) * 1000)} kill {index}"
        if log:
            print(line, file=log, flush=True)

    if options.replay:
        for after, index in read_schedule(options.replay):
            at = started + after
            if not wait_until(nodes, at):
                break
            if 0 <= index < len(nodes) and nodes[index].poll() is None:
                kill(at, index)
            else:
                print(f"kill-nodes.py: the schedule's kill {index} found no "
                      "running node", file=sys.stderr)
    else:
        generator = random.Random(seed)
        at = started
        while len(killed) < options.count:
            at += generator.expovariate(1 / options.mtbf)
            if not wait_until(nodes, at):
                break
            running = [i for i, node in enumerate(nodes)
                       if node.poll() is None and i not in options.spare]
            if running:
                kill(at, running[generator.randrange(len(running))])
    failed = False
    for index, node in enumerate(nodes):
        status = node.wait()
        if index == options.port_from:
            forward.join()
        if index in killed:
            print(f"node {index}: killed")
        else:
            print(f"node {index}: exited with {status}")
            failed = failed or status != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
