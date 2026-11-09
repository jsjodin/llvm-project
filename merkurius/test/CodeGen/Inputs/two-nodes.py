"""Runs a program's nodes against each other over loopback.

Usage: two-nodes.py PROGRAM [SCENARIO...]

With no scenarios, it runs process-remote.m's: pingpong, stall,
simultaneous and cookie. first-slice is omp-target-two-nodes.m's.

Each node is the program with a role as its argument. A node that listens
prints "listening PORT" first, and the driver passes the port to the
other. Every node runs with a tick time of one second, a cookie in
MERK_COOKIE, and HOME a temporary directory, so that no node reads or makes
the user's ~/.merkurius.cookie.

- pingpong: a server, and a client that connects to it, looks names up,
  sends to it and gets replies, idles for three seconds, and sees the
  server close a mailbox and quit.
- stall: a client connects, the driver stops the server with SIGSTOP, and
  the client's lookup must fail as the connection is lost, within a few
  tick times.
- simultaneous: two peers connect to each other at once, exchange
  messages both ways, and must each end with one connection; repeated, as
  which connection wins is a race.
- cookie: a node with another cookie is refused, and one with the same is
  not; without MERK_COOKIE, ~/.merkurius.cookie is made, readable by its
  owner only, and used; a file that others can read, an empty one, an
  empty MERK_COOKIE and no HOME stop a node from listening.
- first-slice: a partner node listens, and a region node connects to it
  and exchanges with it from a target region, which may run on a GPU.
- both-regions: first-slice with the partner answering from a target
  region too, so two GPUs, or one used by two processes, exchange messages.

The output is each node's, prefixed with its role, apart from the ports.
"""

import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

PROGRAM = sys.argv[1]
HOME = tempfile.mkdtemp()
ENVIRONMENT = dict(os.environ, MERK_TICK_TIME="1", MERK_COOKIE="the cookie",
                   HOME=HOME)
LIMIT = 120
ROUNDS = 20
PEER = ["connected", "100 pongs in order", "connections: 1", "bye"]


def start(*arguments, environment=None):
    node = subprocess.Popen(
        [PROGRAM, *arguments],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment or ENVIRONMENT,
    )
    watchdog = threading.Timer(LIMIT, node.kill)
    watchdog.daemon = True
    watchdog.start()
    return node


def port_of(node, role):
    line = node.stdout.readline()
    if not line.startswith("listening "):
        node.kill()
        _, error = node.communicate()
        print(f"{role} did not listen: {line!r} {error!r}")
        sys.exit(1)
    return line.split()[1]


def finish(node, role):
    output, error = node.communicate(timeout=LIMIT)
    if node.returncode != 0 or error:
        print(f"{role} exited with {node.returncode}: {error!r}")
    return output.splitlines()


def pingpong():
    print("pingpong:")
    server = start("server")
    port = port_of(server, "server")
    client = start("client", port)
    for line in finish(client, "client"):
        print("client: " + line)
    print("server: listening")
    for line in finish(server, "server"):
        print("server: " + line)


def stall():
    print("stall:")
    server = start("server")
    port = port_of(server, "server")
    client = start("stall", port)
    print("client: " + client.stdout.readline().rstrip("\n"))
    server.send_signal(signal.SIGSTOP)
    # kill returns before all of the server's threads have stopped: the one
    # that takes the signal stops the others once it runs, and until then
    # they can still answer.
    os.waitpid(server.pid, os.WUNTRACED)
    began = time.monotonic()
    client.stdin.write("go\n")
    client.stdin.flush()
    lines = finish(client, "client")
    took = time.monotonic() - began
    server.send_signal(signal.SIGCONT)
    server.kill()
    server.communicate()
    for line in lines:
        print("client: " + line)
    within = "yes" if 0.5 <= took <= 10 else f"no, {took:.2f} seconds"
    print("lost within 0.5 to 10 seconds: " + within)


def simultaneous():
    print("simultaneous:")
    for round in range(ROUNDS):
        a = start("peer")
        b = start("peer")
        port_a = port_of(a, "a")
        port_b = port_of(b, "b")
        a.stdin.write(port_b + "\n")
        b.stdin.write(port_a + "\n")
        a.stdin.flush()
        b.stdin.flush()
        lines_a = finish(a, "a")
        lines_b = finish(b, "b")
        if lines_a != PEER or lines_b != PEER:
            print(f"round {round}: a {lines_a} b {lines_b}")
            return
    print(f"{ROUNDS} rounds as expected")


def run(role, *arguments, **changes):
    """A node's output on one line, with HOME's path written as HOME."""
    environment = dict(ENVIRONMENT)
    for name, value in changes.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    lines = finish(start(role, *arguments, environment=environment), role)
    return " ".join(lines).replace(HOME, "HOME")


def stop(node):
    node.kill()
    node.communicate()


def cookie():
    print("cookie:")
    server = start("server")
    port = port_of(server, "server")
    print("another cookie: " + run("probe", port, MERK_COOKIE="another"))
    print("the same cookie: " + run("probe", port))
    stop(server)
    # The file, made by the first node that needs it.
    path = os.path.join(HOME, ".merkurius.cookie")
    print("no file, listen: " + run("listen", MERK_COOKIE=None))
    mode = stat.S_IMODE(os.stat(path).st_mode)
    with open(path) as file:
        made = file.read()
    print("made: mode %o, 20 capital letters: %s" %
          (mode, "yes" if re.fullmatch("[A-Z]{20}", made) else repr(made)))
    environment = dict(ENVIRONMENT)
    del environment["MERK_COOKIE"]
    server = start("server", environment=environment)
    port = port_of(server, "server")
    print("the file's cookie: " + run("probe", port, MERK_COOKIE=None))
    print("it in MERK_COOKIE: " + run("probe", port, MERK_COOKIE=made))
    stop(server)
    # Its first line, without the spaces after it.
    os.chmod(path, 0o600)
    with open(path, "w") as file:
        file.write("first line \t\nsecond line\n")
    server = start("server", environment=environment)
    port = port_of(server, "server")
    print("first line: " + run("probe", port, MERK_COOKIE="first line"))
    print("both lines: " +
          run("probe", port, MERK_COOKIE="first line\nsecond line"))
    stop(server)
    os.chmod(path, 0o640)
    print("readable by its group: " + run("listen", MERK_COOKIE=None))
    os.chmod(path, 0o600)
    with open(path, "w") as file:
        file.write("\n")
    print("empty: " + run("listen", MERK_COOKIE=None))
    print("empty MERK_COOKIE: " + run("listen", MERK_COOKIE=""))
    print("no HOME: " + run("listen", MERK_COOKIE=None, HOME=None))


def first_slice():
    print("first-slice:")
    partner = start("partner")
    port = port_of(partner, "partner")
    region = start("region", port)
    for line in finish(region, "region"):
        print("region: " + line)
    print("partner: listening")
    for line in finish(partner, "partner"):
        print("partner: " + line)


def both_regions():
    print("both-regions:")
    partner = start("partner-region")
    port = port_of(partner, "partner-region")
    region = start("region", port)
    for line in finish(region, "region"):
        print("region: " + line)
    print("partner: listening")
    for line in finish(partner, "partner-region"):
        print("partner: " + line)


SCENARIOS = {
    "pingpong": pingpong,
    "stall": stall,
    "simultaneous": simultaneous,
    "cookie": cookie,
    "first-slice": first_slice,
    "both-regions": both_regions,
}

try:
    for name in sys.argv[2:] or ["pingpong", "stall", "simultaneous", "cookie"]:
        SCENARIOS[name]()
finally:
    shutil.rmtree(HOME)
