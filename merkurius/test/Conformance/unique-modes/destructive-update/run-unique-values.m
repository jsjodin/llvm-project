% MANUAL: Destructive update
% uo creates a unique value, ui inspects one and keeps it unique, as
% array.size does, and di destroys one, which the callee may reuse, as
% array.set does. unique(...) is like bound(...), with one reference.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:counter(12)
% OUTPUT-NEXT:3
% OUTPUT-NEXT:[0, 5, 7]
% OUTPUT-NOT:{{.}}

:- module run_unique_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- import_module array, int, list.

:- type counter
    --->    counter(int).

:- inst unique_counter == unique(counter(ground)).

:- pred new_counter(counter::uo) is det.
new_counter(counter(0)).

:- pred step(counter::di, counter::uo) is det.
step(counter(N), counter(N + 1)).

:- pred bump(counter::di(unique_counter), counter::out(unique_counter))
    is det.
bump(counter(N), counter(N + 10)).

main(!IO) :-
    new_counter(C0),
    step(C0, C1),
    step(C1, C2),
    bump(C2, C3),
    io.write_line(C3, !IO),
    array.init(3, 0, A0),
    array.set(1, 5, A0, A1),
    array.size(A1, Size),
    io.write_line(Size, !IO),
    array.set(2, 7, A1, A2),
    array.to_list(A2, L),
    io.write_line(L, !IO).
