% MANUAL: The ``state-of-the-world'' type
% io.state is the state of the world: each I/O predicate takes the only
% reference to it and gives back a unique new one, threaded through calls by
% state variables and explicitly, so the output comes in the order of the
% threading.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:first
% OUTPUT-NEXT:3
% OUTPUT-NEXT:2
% OUTPUT-NEXT:1
% OUTPUT-NEXT:last
% OUTPUT-NOT:{{.}}

:- module run_io_state.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(IO0, IO) :-
    io.write_string("first\n", IO0, IO1),
    count(3, IO1, IO2),
    io.write_string("last\n", IO2, IO).

:- pred count(int::in, io.state::di, io.state::uo) is det.

count(N, !IO) :-
    ( if N > 0 then
        io.write_int(N, !IO),
        io.nl(!IO),
        count(N - 1, !IO)
    else
        true
    ).
