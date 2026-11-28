% DESIGN: Sized floating point types
% float64 is an equivalence type for float (library/float64.m): a float64
% value is a float, so float's operators, conversions and comparisons apply
% to it, it unifies with a float, and io.write writes it as a float. mmc has
% no float64; oracle.py gives it the stand-in
% utils/conformance/emulation/float64.m, the same equivalence.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:4.0
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:[1.5, 2.25, 0.125]
% OUTPUT-NEXT:3
% OUTPUT-NEXT:{1.5, 4.0}
% OUTPUT-NOT:{{.}}

:- module run_float64.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module bool, float, float64, int, list.

:- func half(float64) = float64.
half(X) = X / 2.0.

:- func area(float) = float.
area(R) = R * R.

main(!IO) :-
    X : float64 = 8.0,
    io.write_float(half(X), !IO),
    io.nl(!IO),
    Y : float = half(X),
    io.write(( if Y = 4.0 then yes else no ) : bool, !IO),
    io.nl(!IO),
    Zs : list(float64) = [1.5, 2.25, 0.125],
    io.write(Zs, !IO),
    io.nl(!IO),
    io.write_int(float.truncate_to_int(area(half(3.5))), !IO),
    io.nl(!IO),
    P : {float64, float} = {1.5, area(half(4.0))},
    io.write(P, !IO),
    io.nl(!IO).
