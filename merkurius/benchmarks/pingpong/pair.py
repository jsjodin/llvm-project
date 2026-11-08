"""Runs pingpong's node 2 and node 1 against each other over loopback.

Usage: pair.py PROGRAM WHERE ROUNDS [PARTNER]

WHERE is cpu or gpu, node 1's side. PARTNER is node 2's, partner by default
or partner-gpu, which answers from a target region. Node 2 prints its port
first, which node 1 is given. Both run with a cookie in MERK_COOKIE and
HOME a temporary directory, so that neither reads or makes the user's
~/.merkurius.cookie. Node 1's output is printed.
"""

import os
import shutil
import subprocess
import sys
import tempfile

PROGRAM, WHERE, ROUNDS = sys.argv[1:4]
PARTNER = sys.argv[4] if len(sys.argv) > 4 else "partner"
HOME = tempfile.mkdtemp()
ENVIRONMENT = dict(os.environ, MERK_COOKIE="pingpong", HOME=HOME)
LIMIT = 600

try:
    partner = subprocess.Popen([PROGRAM, PARTNER], stdout=subprocess.PIPE,
                               text=True, env=ENVIRONMENT)
    line = partner.stdout.readline()
    if not line.startswith("listening "):
        partner.kill()
        sys.exit(f"the partner did not listen: {line!r}")
    node1 = subprocess.run([PROGRAM, WHERE, line.split()[1], ROUNDS],
                           env=ENVIRONMENT, timeout=LIMIT)
    partner.wait(timeout=LIMIT)
    sys.exit(node1.returncode or partner.returncode)
finally:
    shutil.rmtree(HOME)
