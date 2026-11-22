% Printing an open term waits for step 6 (docs/INPROGRESS.md, Open types),
% which walks the open layout and numbers unbound variables. Until then it
% stops with a message, rather than walking open(T)'s descriptor, which has
% no functors, as an ordinary term's.
%
% RUN: merkc -fopen-types %s -o %t && not %t 2>&1 | FileCheck %s
% CHECK: before
% CHECK-NEXT: Merkurius runtime: printing an open term is not supported yet

:- module open_write_not_yet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open.

main(!IO) :-
    io.write_string("before\n", !IO),
    io.write(to_open([1, 2]), !IO),
    io.nl(!IO).
