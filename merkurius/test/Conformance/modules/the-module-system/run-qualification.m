% MANUAL: The module system
% Module qualification: an import in the interface for a type used there,
% names imported with use_module written fully qualified, the alternative
% qualifier __, and an end_module declaration naming the module.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 3]
% OUTPUT-NEXT:SHOUT
% OUTPUT-NEXT:[2]
% OUTPUT-NOT:{{.}}

:- module run_qualification.
:- interface.
:- import_module io, list.

:- pred main(io.state::di, io.state::uo) is det.

:- pred lengths(list(string)::in, list(int)::out) is det.

:- implementation.
:- use_module string.

lengths([], []).
lengths([S | Ss], [string.length(S) | Ns]) :-
    lengths(Ss, Ns).

main(!IO) :-
    lengths(["a", "bcd"], Ns),
    io.write_line(Ns, !IO),
    io.write_string(string__to_upper("shout"), !IO),
    io__nl(!IO),
    run_qualification.lengths(["xy"], Ms),
    io.write_line(Ms, !IO).

:- end_module run_qualification.
