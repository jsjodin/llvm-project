% MANUAL: The standard ordering
% compare/3's ordering: numerical for the integer types and float, where
% the two zeros are equal and the infinities are beyond every finite value;
% code points for char; the first argument most significant for tuples;
% and for a discriminated union the constructors' order in the definition,
% then the arguments, the first most significant.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:<=>
% OUTPUT-NEXT:<>>
% OUTPUT-NEXT:<=><
% OUTPUT-NEXT:<<>
% OUTPUT-NEXT:<<=
% OUTPUT-NEXT:><
% OUTPUT-NEXT:<><><<
% OUTPUT-NEXT:<>
% OUTPUT-NEXT:[red, green(2), green(5), blue(1, "x"), amber]
% OUTPUT-NOT:{{.}}

:- module run_ordering.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, int8, list, string, uint, uint64.

:- type colour
    --->    red
    ;       green(int)
    ;       blue(int, string)
    ;       amber.

:- pred show(comparison_result::in, io::di, io::uo) is det.
show(R, !IO) :-
    ( R = (<), io.write_string("<", !IO)
    ; R = (=), io.write_string("=", !IO)
    ; R = (>), io.write_string(">", !IO)
    ).

:- pred cmp(T::in, T::in, io::di, io::uo) is det.
cmp(X, Y, !IO) :-
    compare(R, X, Y),
    show(R, !IO).

main(!IO) :-
    cmp(1, 2, !IO), cmp(-5, -5, !IO), cmp(int.max_int : int, int.min_int : int, !IO),
    io.nl(!IO),
    cmp(-1i8, 1i8, !IO), cmp(3u, 2u, !IO), cmp(uint64.max_uint64 : uint64, 0u64, !IO),
    io.nl(!IO),
    cmp(1.5, 2.5, !IO), cmp(0.0, -0.0, !IO), cmp(float.infinity, 1.0e300, !IO),
    cmp(-float.infinity, -1.0e300, !IO),
    io.nl(!IO),
    cmp('a', 'b', !IO), cmp('Z', 'a', !IO), cmp('é', 'z', !IO),
    io.nl(!IO),
    cmp("abc", "abd", !IO), cmp("", "a", !IO), cmp("b", "b", !IO),
    io.nl(!IO),
    cmp({1, "b"}, {1, "a"}, !IO), cmp({0, "z"}, {1, "a"}, !IO),
    io.nl(!IO),
    cmp(red, amber, !IO), cmp(amber, green(9), !IO),
    cmp(green(1), green(2), !IO), cmp(blue(2, "a"), blue(1, "z"), !IO),
    cmp(blue(1, "a"), blue(1, "b"), !IO), cmp(green(100), blue(0, ""), !IO),
    io.nl(!IO),
    cmp([1, 2], [1, 2, 0], !IO), cmp([3], [1, 2], !IO),
    io.nl(!IO),
    list.sort([amber, blue(1, "x"), green(5), red, green(2)], Sorted),
    io.write_line(Sorted, !IO).
