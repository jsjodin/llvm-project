% MANUAL: Creating higher-order terms
% A predicate with more than one mode cannot be made a higher-order term by
% its name alone: a lambda expression must say which mode.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_multi_mode_name.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- pred app(list(int), list(int), list(int)).
:- mode app(in, in, out) is det.
:- mode app(out, out, in) is multi.
app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :-
    app(Xs, Ys, Zs).

main(!IO) :-
    P = app,                                        % ERROR
    P([1], [2], L),
    io.write_line(L, !IO).
