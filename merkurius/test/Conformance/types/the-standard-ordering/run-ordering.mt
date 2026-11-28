% MANUAL: The standard ordering
% DESIGN: Sized floating point types
% compare/3's ordering: numerical for the integer types and float, where
% the two zeros are equal and the infinities are beyond every finite value;
% code points for char; the first argument most significant for tuples;
% and for a discriminated union the constructors' order in the definition,
% then the arguments, the first most significant. It is a template, run for
% float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_ordering_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_ordering_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:<=>
% OUTPUT-FLOAT-NEXT:<>>
% OUTPUT-FLOAT-NEXT:<=><
% OUTPUT-FLOAT-NEXT:<<>
% OUTPUT-FLOAT-NEXT:<<=
% OUTPUT-FLOAT-NEXT:><
% OUTPUT-FLOAT-NEXT:<><><<
% OUTPUT-FLOAT-NEXT:<>
% OUTPUT-FLOAT-NEXT:[red, green(2), green(5), blue(1, "x"), amber]
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_ordering_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_ordering_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:<=>
% OUTPUT-FLOAT32-NEXT:<>>
% OUTPUT-FLOAT32-NEXT:<=><
% OUTPUT-FLOAT32-NEXT:<<>
% OUTPUT-FLOAT32-NEXT:<<=
% OUTPUT-FLOAT32-NEXT:><
% OUTPUT-FLOAT32-NEXT:<><><<
% OUTPUT-FLOAT32-NEXT:<>
% OUTPUT-FLOAT32-NEXT:[red, green(2), green(5), blue(1, "x"), amber]
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_ordering.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, int8, list, string, uint, uint64.

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

:- func negative_infinity = @T@.
negative_infinity = -@T@.infinity.

:- pred cmp(T::in, T::in, io::di, io::uo) is det.
cmp(X, Y, !IO) :-
    compare(R, X, Y),
    show(R, !IO).

main(!IO) :-
    cmp(1, 2, !IO), cmp(-5, -5, !IO), cmp(int.max_int : int, int.min_int : int, !IO),
    io.nl(!IO),
    cmp(-1i8, 1i8, !IO), cmp(3u, 2u, !IO), cmp(uint64.max_uint64 : uint64, 0u64, !IO),
    io.nl(!IO),
    cmp(1.5@S@, 2.5@S@, !IO), cmp(0.0@S@, -0.0@S@, !IO),
    cmp(@T@.infinity : @T@, 1.0e30@S@, !IO),
    cmp(negative_infinity, -1.0e30@S@, !IO),
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
