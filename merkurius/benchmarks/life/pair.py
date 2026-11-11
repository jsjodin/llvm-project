"""Runs life_two's node 2 and node 1 against each other over loopback.

Usage: pair.py PROGRAM

Node 2, `PROGRAM partner`, prints its port first, which node 1,
`PROGRAM first PORT`, is given. Both run with a cookie in MERK_COOKIE and
HOME a temporary directory, so that neither reads or makes the user's
~/.merkurius.cookie. Node 1's output is printed, then node 2's after it if
it printed more than its port. LIFE_SIZE and LIFE_STEPS pass through.
"""

import os
import shutil
import subprocess
import sys
import tempfile

PROGRAM = sys.argv[1]
HOME = tempfile.mkdtemp()
ENVIRONMENT = dict(os.environ, MERK_COOKIE="life", HOME=HOME)
LIMIT = 600

try:
    partner = subprocess.Popen([PROGRAM, "partner"], stdout=subprocess.PIPE,
                               text=True, env=ENVIRONMENT)
    line = partner.stdout.readline()
    if not line.startswith("listening "):
        partner.kill()
        sys.exit(f"node 2 did not listen: {line!r}")
    try:
        node1 = subprocess.run([PROGRAM, "first", line.split()[1]],
                               env=ENVIRONMENT, timeout=LIMIT)
        rest, _ = partner.communicate(timeout=LIMIT)
    finally:
        if partner.poll() is None:
            partner.kill()
    sys.stdout.flush()
    if rest:
        sys.stdout.write(rest)
    sys.exit(node1.returncode or partner.returncode)
finally:
    shutil.rmtree(HOME)
