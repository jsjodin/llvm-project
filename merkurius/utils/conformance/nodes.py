#!/usr/bin/env python3
"""Runs a conformance test's nodes against each other over loopback.

Usage: nodes.py [--erlang] PROGRAM ROLE...

A distribution test (`% NODES:` in its header, utils/conformance/oracle.py)
is one program run once for each of its roles, each a node. The roles start
in the order given, each as `PROGRAM ROLE ADDRESS...`, with the addresses of
the roles before it that listen. A role that listens prints
`listening ADDRESS` as its first line; every role but the last must print a
line first, which the driver waits for before it starts the next, so that a
role can connect to the ones before it.

Once every role has ended, the driver prints each role's output, standard
error merged into standard output, each line prefixed with the role, in
the order of the roles, without the `listening` lines, and a role's exit
status if it is not 0. The output does not depend on how the nodes'
lines interleave in time, so a test checks what each node saw.

The nodes run with a tick time of one second (MERK_TICK_TIME), a cookie in
MERK_COOKIE, and HOME a temporary directory, so that no node reads or makes
the user's ~/.merkurius.cookie.

With --erlang, PROGRAM is the test's Erlang translation, an escript run in
its own directory. The driver starts an EPMD of its own on a free port
(ERL_EPMD_PORT) and stops it at the end, so that the nodes need no daemon
of the machine's and runs in parallel do not see each other's names; the
translation names each node by its role (Inputs/nodes.hrl).
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

# How long a role may take to print its first line, and all of them to end.
FIRST_LINE_SECONDS = 30
END_SECONDS = 120


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Node:
    def __init__(self, role, command, cwd, env):
        self.role = role
        self.lines = []
        self.first = threading.Event()
        self.process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        for raw in self.process.stdout:
            self.lines.append(raw.decode("utf-8", "replace").rstrip("\n"))
            self.first.set()
        self.first.set()

    def address(self):
        if self.lines and self.lines[0].startswith("listening "):
            return self.lines[0][len("listening "):]
        return None


def main():
    arguments = sys.argv[1:]
    erlang = arguments[:1] == ["--erlang"]
    if erlang:
        arguments = arguments[1:]
    if len(arguments) < 2:
        sys.exit(__doc__)
    program, roles = arguments[0], arguments[1:]
    home = tempfile.mkdtemp()
    env = dict(os.environ, MERK_TICK_TIME="1", MERK_COOKIE="conformance",
               HOME=home)
    cwd = None
    command = [os.path.abspath(program)]
    epmd = None
    if erlang:
        env["ERL_EPMD_PORT"] = str(free_port())
        epmd = subprocess.Popen(["epmd", "-port", env["ERL_EPMD_PORT"]],
                                env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        # The nodes register with it as they start.
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1",
                                 int(env["ERL_EPMD_PORT"]))) == 0:
                    break
            time.sleep(0.05)
        cwd = os.path.dirname(os.path.abspath(program))
        command = ["escript", os.path.basename(program)]
    nodes = []
    try:
        for i, role in enumerate(roles):
            addresses = [n.address() for n in nodes if n.address()]
            node = Node(role, command + [role] + addresses, cwd, env)
            nodes.append(node)
            if i + 1 < len(roles) and not node.first.wait(FIRST_LINE_SECONDS):
                raise RuntimeError(role + " printed nothing")
        deadline = time.monotonic() + END_SECONDS
        for node in nodes:
            try:
                node.process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                node.process.kill()
                node.process.wait()
                node.lines.append("(timed out)")
            node.reader.join()
    finally:
        for node in nodes:
            if node.process.poll() is None:
                node.process.kill()
        if epmd:
            epmd.kill()
            epmd.wait()
        shutil.rmtree(home, ignore_errors=True)
    for node in nodes:
        for line in node.lines[1 if node.address() else 0:]:
            print(node.role + ": " + line)
        if node.process.returncode != 0:
            print("%s exited with %d" % (node.role, node.process.returncode))


if __name__ == "__main__":
    main()
