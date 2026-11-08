% A process that waits for input or for the network holds up no worker
% (docs/INPROGRESS.md, The scheduler, step 7). With one worker, another process
% runs while main waits, and writes first.
%
% RUN: merkc -O2 %s -o %t
% RUN: sh -c "sleep 1; echo hi" | env MERK_WORKERS=1 %t read | FileCheck %s --check-prefix=READ --match-full-lines --strict-whitespace
% RUN: echo hi > %t.in
% RUN: env MERK_WORKERS=1 %t read < %t.in | FileCheck %s --check-prefix=READY --match-full-lines --strict-whitespace
% RUN: %python %S/Inputs/silent-listener.py %t | FileCheck %s --check-prefix=CONNECT --match-full-lines --strict-whitespace

:- module process_blocking.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe, process, string.

:- pred other(mailbox(int)::in, io::di, io::uo) is det.
other(_, !IO) :-
    io.write_string("the other process ran\n", !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    spawn(other, _, !IO),
    ( if Args = ["read"] then
        io.read_line_as_string(Got, !IO),
        ( if Got = ok(Line) then
            io.write_string("read " ++ Line, !IO)
        else
            io.write_string("no line\n", !IO)
        )
    else if Args = ["connect", Address] then
        connect(Address, Result, !IO),
        (
            Result = ok(_),
            io.write_string("connected\n", !IO)
        ;
            Result = error(_),
            io.write_string("connect failed\n", !IO)
        )
    else
        io.write_string("usage\n", !IO)
    ).

% READ:the other process ran
% READ-NEXT:read hi
% READ-NOT:{{.}}

% Input that is there needs no wait, so main does not move, and ends first.
% READY:read hi
% READY-NOT:{{.}}

% CONNECT:the other process ran
% CONNECT-NEXT:connect failed
% CONNECT-NOT:{{.}}
