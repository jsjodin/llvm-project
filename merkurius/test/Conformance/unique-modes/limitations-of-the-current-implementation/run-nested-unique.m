% MANUAL: Limitations of the current implementation
% The manual says nested unique modes cannot be used, but mmc accepts them,
% and merkc follows it (design document, "Where mmc and the manual
% disagree", "Nested unique modes"). A unique field of a unique term is
% taken out and destroyed. Deconstructing a ui argument is the limitation
% that is real (differs-deconstruct-ui.m).
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[4, 0]
% OUTPUT-NOT:{{.}}

:- module run_nested_unique.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- import_module array, list.

:- type holder
    --->    holder(array(int)).

:- inst unique_holder == unique(holder(uniq_array)).

:- pred make(holder::out(unique_holder)) is det.
make(holder(A)) :-
    array.init(2, 0, A).

:- pred update(holder::di(unique_holder), holder::out(unique_holder))
    is det.
update(holder(A0), holder(A)) :-
    array.set(0, 4, A0, A).

main(!IO) :-
    make(H0),
    update(H0, H1),
    H1 = holder(A),
    array.to_list(A, L),
    io.write_line(L, !IO).
