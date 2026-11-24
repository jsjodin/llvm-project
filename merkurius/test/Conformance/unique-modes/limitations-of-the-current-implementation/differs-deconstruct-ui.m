% MANUAL: Limitations of the current implementation
% DIFFERS: implementation: mmc cannot deconstruct a ui argument: its fields
% become ground, so the argument no longer matches unique at the end, the
% limitation the manual describes. merkc keeps the argument unique.
% Reading a unique value's field through ui.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{1, 2}
% OUTPUT-NOT:{{.}}
% MMC-STATUS: rejected
% MMC: 31: In clause for `peek(ui, out)': mode error: argument 1 did not get sufficiently instantiated. Final instantiatedness of `HeadVar__1' was `unique(counter(ground))', expected final instantiatedness was `unique'.

:- module differs_deconstruct_ui.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- type counter
    --->    counter(int).

:- pred new_counter(counter::uo) is det.
new_counter(counter(1)).

:- pred peek(counter::ui, int::out) is det.
peek(counter(N), N).

:- pred step(counter::di, counter::uo) is det.
step(counter(N), counter(N + 1)).

main(!IO) :-
    new_counter(C0),
    peek(C0, N0),
    step(C0, C1),
    peek(C1, N1),
    io.write_line({N0, N1}, !IO).
