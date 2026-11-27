# Receives between the host's processes

How long a message takes through a receive goal, or through receive/5, from
main's first send to
the consumer's answer, over many messages: throughput, with main and the
consumer running at once.

| Case | What it measures |
| --- | --- |
| `plain N` | One mailbox, every message selected |
| `two N` | Two mailboxes of the consumer's, the messages alternating |
| `skip K N` | K messages that no case selects wait in front of the N, so each receive looks past them |
| `any N` | receive/5 with `any`, every message taken |
| `selector N` | receive/5 with a named selector, every message selected |
| `selector-skip K N` | receive/5 with the selector, K messages that it rejects in front of the N |

`run.sh [build-dir] [program]` builds the program and runs the cases, or
runs a copy built before a change. `RECEIVE_GOAL_MESSAGES` sets the
messages, 1000000 by default.

The numbers are in the design document's Measurements section, "Receive
goals on the host" and "Plain receives on the host".
