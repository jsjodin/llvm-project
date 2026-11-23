% The library's errors in a GPU process: require's software errors and the
% domain errors of float and math, whose foreign code throws through the
% runtime's merk_library_error, call the device runtime's
% merk_device_uncaught_exception in a GPU's compile, an uncaught exception,
% which the host reports as it reports its own. It ends only the process,
% back in its kernel's slot loop (docs/INPROGRESS.md, Exceptions, step 3),
% and a monitor sees it failed with the report's text. This file runs the
% same calls on the host, where they end the program, and whose reports
% process-library-errors-gpu.test checks the GPU's against; the call that
% does not fail prints its result.
%
% RUN: merkc -fprocess-declarations %s -o %t
% RUN: %t host ok | FileCheck %s --check-prefix=OK --match-full-lines --strict-whitespace
% RUN: not %t host unexpected 2>&1 | FileCheck %s --check-prefix=UNEXPECTED --match-full-lines --strict-whitespace
% RUN: not %t host sqrt 2>&1 | FileCheck %s --check-prefix=SQRT --match-full-lines --strict-whitespace
% RUN: not %t host pow 2>&1 | FileCheck %s --check-prefix=POW --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %t2 host ok | FileCheck %s --check-prefix=OK --match-full-lines --strict-whitespace
% RUN: not %t2 host unexpected 2>&1 | FileCheck %s --check-prefix=UNEXPECTED --match-full-lines --strict-whitespace
% RUN: not %t2 host sqrt 2>&1 | FileCheck %s --check-prefix=SQRT --match-full-lines --strict-whitespace
% RUN: not %t2 host pow 2>&1 | FileCheck %s --check-prefix=POW --match-full-lines --strict-whitespace
%
% OK:1.5
% UNEXPECTED:Uncaught Merkurius exception:
% UNEXPECTED-NEXT:Software Error: process_library_errors: predicate `process_library_errors.compute'/3: Unexpected: no such operation
% UNEXPECTED-NOT:{{.}}
% SQRT:Uncaught Merkurius exception:
% SQRT-NEXT:domain_error("math.sqrt")
% SQRT-NOT:{{.}}
% POW:Uncaught Merkurius exception:
% POW-NEXT:domain_error("float.pow: zero base")
% POW-NOT:{{.}}
% DOWN-UNEXPECTED:failed: Software Error: process_library_errors: predicate `process_library_errors.compute'/3: Unexpected: no such operation
% DOWN-UNEXPECTED-NOT:{{.}}
% DOWN-SQRT:failed: domain_error("math.sqrt")
% DOWN-SQRT-NOT:{{.}}
% DOWN-POW:failed: domain_error("float.pow: zero base")
% DOWN-POW-NOT:{{.}}

:- module process_library_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, math, process, require, string.

    % Answers each number it receives with the operation's result on it.
:- process worker(int, mailbox(float)) receives float on gpu.

worker(Op, Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(X),
        compute(Op, X, Y),
        send(Reply, Y, !IO),
        worker(Op, Reply, Self, !IO)
    ;
        R = timeout
    ).

    % Operation 0 is the square root, 1 a power of zero to the number, and
    % 2 none.
:- pred compute(int::in, float::in, float::out) is det.
:- pragma no_inline(compute/3).

compute(Op, X, Y) :-
    ( if Op = 0 then
        Y = math.sqrt(X)
    else if Op = 1 then
        Y = float.pow(0.0, float.truncate_to_int(X))
    else
        unexpected($module, $pred, "no such operation")
    ).

:- func op(string) = int.

op(Name) =
    ( if Name = "ok" then 0
    else if Name = "sqrt" then 0
    else if Name = "pow" then 1
    else 2 ).

:- func argument(string) = float.

argument(Name) =
    ( if Name = "ok" then 2.25
    else if Name = "sqrt" then -1.0
    else -1.0 ).

:- func reason(down_info) = string.

reason(down_info(_, normal)) = "normal".
reason(down_info(_, noproc)) = "noproc".
reason(down_info(_, killed)) = "killed".
reason(down_info(_, kernel_ended)) = "kernel_ended".
reason(down_info(_, noconnection)) = "noconnection".
reason(down_info(_, failed(Text))) = "failed: " ++ Text.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [Where, Name] then
        ( if Where = "gpu" then
            new_mailbox(Me, !IO),
            new_mailbox(Downs, !IO),
            spawn(gpu(0), worker(op(Name), Me), W, !IO),
            monitor(W, Downs, reason, _, !IO),
            send(W, argument(Name), !IO),
            ( if Name = "ok" then
                receive(Me, any, infinity, R, !IO),
                ( if R = message(Y) then
                    io.write_float(Y, !IO),
                    io.nl(!IO)
                else
                    true
                )
            else
                receive(Downs, any, infinity, D, !IO),
                ( if D = message(Text) then
                    io.write_string(Text ++ "\n", !IO)
                else
                    true
                )
            )
        else
            compute(op(Name), argument(Name), Y),
            io.write_float(Y, !IO),
            io.nl(!IO)
        )
    else
        io.write_string("usage: process-library-errors host|gpu " ++
            "ok|unexpected|sqrt|pow\n", !IO)
    ).
