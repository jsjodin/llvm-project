% Switches on strings and floats, which are tested case by case: an
% if-then-else for each case, whose condition compares the value with the
% case's constants, and fail after the last, unless the switch is complete,
% as finite/2's, when the last case is what is left. A float is compared
% as ==, so 0.0 is -0.0's case. A case may have more than one constant, as
% number/2's third, and a nondeterministic switch, names/2, gives each of
% its case's solutions. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-LABEL: merk.proc private @switch_strings.finite_2_0(%arg0: !merk.string) -> i64 attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: %[[X:.*]] = merk.ite -> i64 {
% MERK-NEXT: %[[C:.*]] = merk.constant "x" : !merk.string
% MERK-NEXT: %[[EQ:.*]] = merk.string_equal %arg0, %[[C]]
% MERK-NEXT: merk.guard %[[EQ]]
% MERK-NEXT: merk.yield
% MERK-NEXT: } then {
% MERK-NEXT: %[[TEN:.*]] = merk.constant 10 : i64
% MERK-NEXT: merk.yield %[[TEN]] : i64
% MERK-NEXT: } else {
% MERK-NEXT: %[[TWENTY:.*]] = merk.constant 20 : i64
% MERK-NEXT: merk.yield %[[TWENTY]] : i64
% MERK-NEXT: }
% MERK-NEXT: merk.succeed %[[X]] : i64
% MERK-LABEL: merk.proc private @switch_strings.colour_f1_0(%arg0: !merk.string) -> i64 attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: merk.ite -> i64 {
% MERK-NEXT: %[[EMPTY:.*]] = merk.constant "" : !merk.string
% MERK-NEXT: %[[EQ:.*]] = merk.string_equal %arg0, %[[EMPTY]]
% MERK-NEXT: merk.guard %[[EQ]]
% MERK: merk.constant "\C3\A9\E2\82\AC\F0\9F\98\80" : !merk.string
% MERK: } else {
% MERK-NEXT: merk.fail
% MERK-NEXT: }
% MERK-LABEL: merk.proc private @switch_strings.number_2_0(%arg0: !merk.string) -> i64
% MERK-NEXT: merk.ite -> i64 {
% MERK-NEXT: %[[DREI:.*]] = merk.constant "drei" : !merk.string
% MERK-NEXT: %[[EQ1:.*]] = merk.string_equal %arg0, %[[DREI]]
% MERK-NEXT: %[[THREE:.*]] = merk.constant "three" : !merk.string
% MERK-NEXT: %[[EQ2:.*]] = merk.string_equal %arg0, %[[THREE]]
% MERK-NEXT: %[[OR1:.*]] = arith.ori %[[EQ1]], %[[EQ2]] : i1
% MERK-NEXT: %[[TROIS:.*]] = merk.constant "trois" : !merk.string
% MERK-NEXT: %[[EQ3:.*]] = merk.string_equal %arg0, %[[TROIS]]
% MERK-NEXT: %[[OR2:.*]] = arith.ori %[[OR1]], %[[EQ3]] : i1
% MERK-NEXT: merk.guard %[[OR2]]
% MERK-LABEL: merk.proc private @switch_strings.weight_f1_0(%arg0: f64) -> !merk.string
% MERK-NEXT: merk.ite -> !merk.string {
% MERK-NEXT: %[[ZERO:.*]] = merk.constant 0.000000e+00 : f64
% MERK-NEXT: %[[EQ:.*]] = arith.cmpf oeq, %arg0, %[[ZERO]] : f64
% MERK-NEXT: merk.guard %[[EQ]]

% OUTPUT:red: 1
% OUTPUT-NEXT:green: 2
% OUTPUT-NEXT:blue: 3
% OUTPUT-NEXT:: 0
% OUTPUT-NEXT:é€😀: 4
% OUTPUT-NEXT:purple: none
% OUTPUT-NEXT:Red: none
% OUTPUT-NEXT:one: 1
% OUTPUT-NEXT:two: 2
% OUTPUT-NEXT:three: 3
% OUTPUT-NEXT:drei: 3
% OUTPUT-NEXT:trois: 3
% OUTPUT-NEXT:four: none
% OUTPUT-NEXT:["alpha", "ant", "apple"]
% OUTPUT-NEXT:["bravo"]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:half
% OUTPUT-NEXT:one
% OUTPUT-NEXT:minus zero
% OUTPUT-NEXT:minus zero
% OUTPUT-NEXT:big
% OUTPUT-NEXT:none
% OUTPUT-NEXT:{10, 20}

:- module switch_strings.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, solutions, string.

:- func colour(string) = int is semidet.
colour("red") = 1.
colour("green") = 2.
colour("blue") = 3.
colour("") = 0.
colour("é€😀") = 4.

:- pred number(string::in, int::out) is semidet.
number(S, N) :-
    ( S = "one", N = 1
    ; S = "two", N = 2
    ; ( S = "three" ; S = "drei" ; S = "trois" ), N = 3
    ).

:- pred names(string::in, string::out) is nondet.
names("a", "alpha").
names("a", "apple").
names("b", "bravo").
names("a", "ant").

:- func weight(float) = string is semidet.
weight(0.5) = "half".
weight(1.0) = "one".
weight(-0.0) = "minus zero".
weight(1.5e10) = "big".

:- pred finite(string, int).
:- mode finite(in(bound("x" ; "y")), out) is det.
finite("x", 10).
finite("y", 20).

main(!IO) :-
    list.foldl(show_colour, ["red", "green", "blue", "", "é€😀", "purple", "Red"], !IO),
    list.foldl(show_number, ["one", "two", "three", "drei", "trois", "four"], !IO),
    solutions(names("a"), A), io.write(A, !IO), io.nl(!IO),
    solutions(names("b"), B), io.write(B, !IO), io.nl(!IO),
    solutions(names("c"), C), io.write(C, !IO), io.nl(!IO),
    list.foldl(show_weight, [0.5, 1.0, 0.0, -0.0, 15000000000.0, 2.0], !IO),
    finite("x", X), finite("y", Y), io.write({X, Y}, !IO), io.nl(!IO).

:- pred show_colour(string::in, io::di, io::uo) is det.
show_colour(S, !IO) :-
    ( if N = colour(S) then io.write_string(string.format("%s: %d\n", [s(S), i(N)]), !IO)
    else io.write_string(string.format("%s: none\n", [s(S)]), !IO) ).

:- pred show_number(string::in, io::di, io::uo) is det.
show_number(S, !IO) :-
    ( if number(S, N) then io.write_string(string.format("%s: %d\n", [s(S), i(N)]), !IO)
    else io.write_string(string.format("%s: none\n", [s(S)]), !IO) ).

:- pred show_weight(float::in, io::di, io::uo) is det.
show_weight(F, !IO) :-
    ( if W = weight(F) then io.write_string(W ++ "\n", !IO)
    else io.write_string("none\n", !IO) ).
