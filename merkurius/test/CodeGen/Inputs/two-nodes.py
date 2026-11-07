"""Runs process-remote.m's nodes against each other over loopback.

Usage: two-nodes.py PROGRAM

Each node is the program with a role as its argument. A node that listens
prints "listening PORT" first, and the driver passes the port to the
other. Every node runs with a tick time of one second.

- pingpong: a server, and a client that connects to it, looks names up,
  sends to it and gets replies, idles for three seconds, and sees the
  server close a mailbox and quit.
- stall: a client connects, the driver stops the server with SIGSTOP, and
  the client's lookup must fail as the connection is lost, within a few
  tick times.
- simultaneous: two peers connect to each other at once, exchange
  messages both ways, and must each end with one connection; repeated, as
  which connection wins is a race.

The output is each node's, prefixed with its role, apart from the ports.
"""

import os
import signal
import subprocess
import sys
import threading
import time

PROGRAM = sys.argv[1]
ENVIRONMENT = dict(os.environ, MERK_TICK_TIME="1")
LIMIT = 120
ROUNDS = 20
PEER = ["connected", "100 pongs in order", "connections: 1", "bye"]


def start(*arguments):
    node = subprocess.Popen(
        [PROGRAM, *arguments],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ENVIRONMENT,
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


pingpong()
stall()
simultaneous()
