#!/usr/bin/env bash
# Builds the receive-goal benchmark and runs its cases (README.md): one
# mailbox with every message selected, messages that no case selects in
# front of those it does, and two mailboxes; then receive/5's, with any,
# with a selector, and with messages it rejects in front.
#
#   run.sh [build-dir] [program]
#
# build-dir is the Merkurius build, ../../../../build-merk by default.
# program is a receive-goal binary to run instead of building one, as a
# copy built before a change. RECEIVE_GOAL_MESSAGES sets the messages of
# each case, 1000000 by default; the cases with messages in front take a
# tenth and a hundredth of them, as each receive looks past 100 and 1000.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
n="${RECEIVE_GOAL_MESSAGES:-1000000}"
if [[ -n "${2:-}" ]]; then
  program="$2"
else
  out="$(mktemp -d)"
  program="$out/receive-goal"
  "$build/bin/merkc" -O2 -freceive-goal -fprocess-declarations \
    -Wno-unselected-message "$here/receive-goal.m" -o "$program"
fi
"$program" plain "$n"
"$program" two "$n"
"$program" skip 100 "$((n / 10))"
"$program" skip 1000 "$((n / 100))"
"$program" any "$n"
"$program" selector "$n"
"$program" selector-skip 100 "$((n / 10))"
"$program" selector-skip 1000 "$((n / 100))"
