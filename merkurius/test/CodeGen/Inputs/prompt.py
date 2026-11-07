"""Answers a program's prompt: reads its first line, which it must flush
before it reads, then gives it its input and prints all it wrote.

Usage: prompt.py PROGRAM INPUT

INPUT is in Python's escapes, such as \\n, \\x00 and \\xff, as a command
line cannot hold a NUL or bytes that are not UTF-8.

Without the flush the first line would not come until the program exits,
and the program would wait for its input until the timeout.
"""

import codecs
import subprocess
import sys

program = subprocess.Popen(
    [sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    first = program.stdout.readline()
    sys.stdout.buffer.write(b"before the input: " + first)
    output, _ = program.communicate(
        codecs.escape_decode(sys.argv[2])[0], timeout=60)
except subprocess.TimeoutExpired:
    program.kill()
    print("timed out")
    sys.exit(1)
sys.stdout.buffer.write(output)
print("exit status %d" % program.returncode)
