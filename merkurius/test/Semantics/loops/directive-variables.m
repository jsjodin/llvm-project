% A directive's clauses name variables but are no occurrences of them: a
% private clause naming a loop's counter does not make the counter occur
% outside the loop, which would draw the warning that an iterator's
% variable is a different one there.
%
% RUN: merkc -fc1 -fdo-loops -fopenmp -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
:- module directive_variables.
:- interface.
:- pred sum(int::in, int::out) is det.
:- implementation.
:- import_module int.

sum(N, S) :-
    %$omp parallel do private(I, X) reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do X = I * 2, S1 = S0 + X ).
% CHECK: call pred directive_variables.sum__body_16_7/3 mode 0: sum__body_16_7(I, S0, S1) [det]
