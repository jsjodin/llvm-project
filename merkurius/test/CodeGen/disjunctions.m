% Disjunctions with at most one solution as the goals around them see it,
% such as tests, and those where only the first solution is wanted, in a
% committed choice procedure: each branch in turn, as an if-then-else
% whose condition is the branch, tried when the one before it fails, and
% the last branch's call a tail call. A branch that binds variables of its
% own and then fails leaves nothing behind. The output is mmc's for the
% same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module disjunctions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

:- type colour ---> red ; green ; blue.

:- pred mem(int::in, list(int)::in) is semidet.
mem(X, [H | T]) :-
    ( X = H ; mem(X, T) ).

:- pred outside(int::in) is semidet.
outside(X) :-
    ( X > 10 ; X < -5 ; X = 0 ).

% A branch that binds a local and then fails gives way to the next.
:- pred tricky(int::in) is semidet.
tricky(X) :-
    ( Y = X * 2, Y > 100 ; Z = X + 1, Z = 4 ).

:- pred pick(int::out) is cc_multi.
pick(X) :-
    ( X = 1 ; X = 2 ).

% Committed choice: the first branch that succeeds gives the outputs.
:- pred first_big(list(int)::in, int::out, string::out) is cc_nondet.
first_big(L, X, Why) :-
    (
        L = [X | _], X > 5, Why = "first"
    ;
        L = [_, X | _], X > 5, Why = "second"
    ;
        L = [_, _, X | _], Why = "third"
    ).

:- pred warm(colour::in) is semidet.
warm(C) :-
    ( C = red ; C = green, fail ).

:- pred nested(int::in) is semidet.
nested(X) :-
    ( ( X = 1 ; X = 2 ) ; ( X = 3 ; X > 9 ) ).

:- pred report(string::in, bool_t::in, io::di, io::uo) is det.
:- type bool_t ---> yes ; no.
report(What, B, !IO) :-
    io.write_string(What, !IO), io.write_string(": ", !IO), io.write(B, !IO), io.nl(!IO).

:- func t((pred)) = bool_t.
:- mode t((pred) is semidet) = out is det.
t(P) = ( if P then yes else no ).

main(!IO) :-
    report("mem 3", t((pred) is semidet :- mem(3, [1, 2, 3])), !IO),
    report("mem 4", t((pred) is semidet :- mem(4, [1, 2, 3])), !IO),
    report("outside 11", t((pred) is semidet :- outside(11)), !IO),
    report("outside -6", t((pred) is semidet :- outside(-6)), !IO),
    report("outside 0", t((pred) is semidet :- outside(0)), !IO),
    report("outside 3", t((pred) is semidet :- outside(3)), !IO),
    report("tricky 3", t((pred) is semidet :- tricky(3)), !IO),
    report("tricky 60", t((pred) is semidet :- tricky(60)), !IO),
    report("tricky 5", t((pred) is semidet :- tricky(5)), !IO),
    report("warm red", t((pred) is semidet :- warm(red)), !IO),
    report("warm green", t((pred) is semidet :- warm(green)), !IO),
    report("warm blue", t((pred) is semidet :- warm(blue)), !IO),
    report("nested 2", t((pred) is semidet :- nested(2)), !IO),
    report("nested 3", t((pred) is semidet :- nested(3)), !IO),
    report("nested 10", t((pred) is semidet :- nested(10)), !IO),
    report("nested 5", t((pred) is semidet :- nested(5)), !IO),
    pick(P),
    io.write_int(P, !IO), io.nl(!IO),
    ( if first_big([7, 1, 1], X1, W1) then
        io.write({X1, W1}, !IO)
    else
        io.write_string("none", !IO)
    ),
    io.nl(!IO),
    ( if first_big([1, 8, 1], X2, W2) then
        io.write({X2, W2}, !IO)
    else
        io.write_string("none", !IO)
    ),
    io.nl(!IO),
    ( if first_big([1, 1, 2], X3, W3) then
        io.write({X3, W3}, !IO)
    else
        io.write_string("none", !IO)
    ),
    io.nl(!IO),
    ( if first_big([1, 1], X4, W4) then
        io.write({X4, W4}, !IO)
    else
        io.write_string("none", !IO)
    ),
    io.nl(!IO).

% OUTPUT:mem 3: yes
% OUTPUT-NEXT:mem 4: no
% OUTPUT-NEXT:outside 11: yes
% OUTPUT-NEXT:outside -6: yes
% OUTPUT-NEXT:outside 0: yes
% OUTPUT-NEXT:outside 3: no
% OUTPUT-NEXT:tricky 3: yes
% OUTPUT-NEXT:tricky 60: yes
% OUTPUT-NEXT:tricky 5: no
% OUTPUT-NEXT:warm red: yes
% OUTPUT-NEXT:warm green: no
% OUTPUT-NEXT:warm blue: no
% OUTPUT-NEXT:nested 2: yes
% OUTPUT-NEXT:nested 3: yes
% OUTPUT-NEXT:nested 10: yes
% OUTPUT-NEXT:nested 5: no
% OUTPUT-NEXT:1
% OUTPUT-NEXT:{7, "first"}
% OUTPUT-NEXT:{8, "second"}
% OUTPUT-NEXT:{2, "third"}
% OUTPUT-NEXT:none

% MERK-LABEL: merk.proc private @disjunctions.mem_2_0(%arg0: i64, %arg1: !merk.term<"list.list(int)">)
% MERK: merk.ite {
% MERK-NEXT: %[[EQ:.*]] = arith.cmpi eq, %arg0, %{{.*}} : i64
% MERK-NEXT: merk.guard %[[EQ]]
% MERK-NEXT: merk.yield
% MERK-NEXT: } then {
% MERK-NEXT: merk.yield
% MERK-NEXT: } else {
% MERK-NEXT: merk.call @disjunctions.mem_2_0(%arg0, %{{.*}}) : (i64, !merk.term<"list.list(int)">) -> ()
% MERK-NEXT: merk.yield
% MERK-NEXT: }
% MERK-NEXT: merk.succeed
