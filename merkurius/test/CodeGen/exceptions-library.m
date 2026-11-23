% The library's own errors are values a try catches (docs/INPROGRESS.md,
% Exceptions, step 1): code generation's checks, as a division by zero or
% int.abs of min_int, and the library's C, as math.sqrt's, float.pow's and
% require's, give their text to the runtime's merk_library_error, which
% throws the software_error, domain_error or index_out_of_bounds it reads
% there. A comparison of NaNs through a type variable stays a fatal error,
% as in mmc, but one at a known float, as here, has no check. OUTPUT is
% mmc's.
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module exceptions_library.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module array, exception, float, int, list, math, require, string, univ.

:- pred show(exception_result(T)::in, io::di, io::uo) is det.
show(succeeded(_), !IO) :- io.write_string("succeeded\n", !IO).
show(failed, !IO) :- io.write_string("failed\n", !IO).
show(exception(U), !IO) :-
    ( if univ_to_type(U, software_error(S)) then io.format("software_error(%s)\n", [s(S)], !IO)
    else if univ_to_type(U, exception.domain_error(S)) then io.format("domain_error(%s)\n", [s(S)], !IO)
    else if univ_to_type(U, array.index_out_of_bounds(S)) then io.format("index_out_of_bounds(%s)\n", [s(S)], !IO)
    else io.write_string("other: ", !IO), io.write(univ_value(U), !IO), io.nl(!IO) ).

:- pred div0(int::in, int::out) is det.
div0(X, Y) :- Y = 100 // X.
:- pred req(int::in, int::out) is det.
req(X, Y) :- ( if X > 0 then Y = X else error("req msg") ).
:- pred sq(float::in, float::out) is det.
sq(X, Y) :- Y = math.sqrt(X).
:- pred idx(int::in, int::out) is det.
idx(I, Y) :- A = array.from_list([1, 2, 3]), Y = A ^ elem(I).
:- pred nan(float::in, comparison_result::out) is det.
nan(X, R) :- compare(R, X, X).
:- pred ini(int::in, int::out) is det.
ini(N, Y) :- array.init(N, 0, A), Y = array.size(A).
:- pred pw(float::in, float::out) is det.
pw(X, Y) :- Y = float.pow(X, -1).
:- pred unex(int::in, int::out) is det.
unex(X, Y) :- ( if X > 0 then Y = X else unexpected($pred, "bad") ).
:- pred absmin(int::in, int::out) is det.
absmin(X, Y) :- Y = int.abs(X).

main(!IO) :-
    try(div0(0), R1), show(R1, !IO),
    try(req(0), R2), show(R2, !IO),
    try(sq(-1.0), R3), show(R3, !IO),
    try(idx(7), R4), show(R4, !IO),
    Inf = float.max * 10.0, try(nan(Inf - Inf), R5), show(R5, !IO),
    try(ini(-1), R6), show(R6, !IO),
    try(pw(0.0), R7), show(R7, !IO),
    try(unex(0), R8), show(R8, !IO),
    try(absmin(int.min_int), R9), show(R9, !IO).
% OUTPUT:domain_error(int.'//': division by zero)
% OUTPUT-NEXT:software_error(req msg)
% OUTPUT-NEXT:domain_error(math.sqrt)
% OUTPUT-NEXT:index_out_of_bounds(array.lookup: index 7 not in range [0, 2])
% OUTPUT-NEXT:succeeded
% OUTPUT-NEXT:software_error(predicate `array.init'/3: Unexpected: negative size)
% OUTPUT-NEXT:domain_error(float.pow: zero base)
% OUTPUT-NEXT:software_error(predicate `exceptions_library.unex'/2: Unexpected: bad)
% OUTPUT-NEXT:software_error(int.abs: abs(min_int) would overflow)
