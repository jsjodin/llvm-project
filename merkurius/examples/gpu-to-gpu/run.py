"""Runs the demo's two nodes on this machine, over TCP on loopback.

Usage: run.py PROGRAM

Starts node B, reads the port it listens on, and starts node A. Their messages
appear as they happen. For two machines, run the program by hand (README.md).
"""

import os
import re
import subprocess
import sys

PROGRAM = sys.argv[1]
ENVIRONMENT = dict(os.environ, MERK_COOKIE="demo")
LIMIT = 60

node_b = subprocess.Popen([PROGRAM, "b"], stdout=subprocess.PIPE, text=True,
                          env=ENVIRONMENT)
line = node_b.stdout.readline()
print(line, end="", flush=True)
port = re.search(r"port (\d+)", line)
if not port:
    node_b.kill()
    sys.exit(f"node B did not listen: {line!r}")
node_a = subprocess.Popen([PROGRAM, "a", f"127.0.0.1:{port.group(1)}"],
                          env=ENVIRONMENT)
for line in node_b.stdout:
    print(line, end="", flush=True)
sys.exit(node_a.wait(timeout=LIMIT) or node_b.wait(timeout=LIMIT))
