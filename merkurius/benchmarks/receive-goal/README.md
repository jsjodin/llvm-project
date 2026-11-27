# Receive goals between the host's processes

How long a message takes through a receive goal, from main's first send to
the consumer's answer, over many messages: throughput, with main and the
consumer running at once.

| Case | What it measures |
| --- | --- |
| `plain N` | One mailbox, every message selected |
| `two N` | Two mailboxes of the consumer's, the messages alternating |
| `skip K N` | K messages that no case selects wait in front of the N, so each receive looks past them |

`run.sh [build-dir] [program]` builds the program and runs the cases, or
runs a copy built before a change. `RECEIVE_GOAL_MESSAGES` sets the
messages, 1000000 by default.

The numbers are in the design document's Measurements section, "Receive
goals on the host".
