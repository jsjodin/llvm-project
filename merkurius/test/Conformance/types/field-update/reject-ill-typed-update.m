% MANUAL: Field update
% A field whose update would not be well-typed, as either of the manual's
% unsettable's, cannot be updated.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:

:- module reject_ill_typed_update.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type unsettable
    --->    some [T] unsettable(
                unsettable1 :: T,
                unsettable2 :: T
            ).

main(!IO) :-
    U0 = 'new unsettable'(1, 2),
    U = U0 ^ unsettable1 := "one",                  % ERROR
    io.write_line(U, !IO).
