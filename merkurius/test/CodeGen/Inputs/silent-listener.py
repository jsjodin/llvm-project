#!/usr/bin/env python3
# A listener that accepts a connection, says nothing for a second, and closes
# it, so that a connect waits in its handshake. Runs the program named on the
# command line, with one worker, and passes on what it writes.
import os
import socket
import subprocess
import sys
import threading
import time


def main():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    address = "127.0.0.1:%d" % listener.getsockname()[1]

    def silent():
        connection, _ = listener.accept()
        time.sleep(1)
        connection.close()

    threading.Thread(target=silent, daemon=True).start()
    environment = dict(os.environ, MERK_WORKERS="1", MERK_COOKIE="cookie")
    done = subprocess.run(
        [sys.argv[1], "connect", address],
        stdout=subprocess.PIPE,
        timeout=30,
        env=environment,
    )
    sys.stdout.buffer.write(done.stdout)
    return done.returncode


sys.exit(main())
