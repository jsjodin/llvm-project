% The call in a semidet disjunction's last branch is a tail call, so a
% search through five million list elements runs in constant stack at
% every optimization level, as in mmc. The output is mmc's.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module disjunction_tail_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred mem(int::in, list(int)::in) is semidet.
mem(X, [H | T]) :-
    ( X = H ; mem(X, T) ).

:- pred build(int::in, list(int)::in, list(int)::out) is det.
build(N, L0, L) :-
    ( if N = 0 then L = L0 else build(N - 1, [N | L0], L) ).

main(!IO) :-
    build(5000000, [], L),
    ( if mem(4999999, L) then io.write_string("found\n", !IO) else io.write_string("missing\n", !IO) ),
    ( if mem(-1, L) then io.write_string("found\n", !IO) else io.write_string("missing\n", !IO) ).

% CHECK:found
% CHECK-NEXT:missing
