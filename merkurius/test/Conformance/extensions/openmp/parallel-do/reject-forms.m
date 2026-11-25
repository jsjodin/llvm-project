% DESIGN: Loops / Under a parallel directive
% A parallel loop is counted by its one for: a list, a count and a fromto
% that ends the loop are sequential by nature, so each is an error. A
% value carried from one iteration to the next is named in a reduction
% clause, or the loop is an error, and so is one that threads the io
% state. Its body must be det and pure. Each error is the directive's,
% not a silent sequential loop.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopenmp -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
:- module reject_forms.
:- interface.
:- import_module io, list.
:- pred lists(list(int)::in, int::out) is det.
:- pred counted(int::in, int::out) is det.
:- pred ended(list(int)::in, int::out) is semidet.
:- pred unnamed(int::in, int::out) is det.
:- pred state(int::in, int::out) is det.
:- pred printing(int::in, io::di, io::uo) is det.
:- pred failing(int::in, int::out) is semidet.
:- implementation.
:- import_module int.

lists(Xs, S) :-
    %$omp parallel do reduction(+)
    ( foreach(X, Xs), fromto(0, S0, S1, S) do S1 = S0 + X ).
% CHECK: reject-forms.m:[[@LINE-1]]:{{[0-9]+}}: error: a parallel do loop is counted by its for, so it cannot have foreach
counted(N, S) :-
    %$omp parallel do reduction(+)
    ( for(I, 1, N), count(_, 1, _), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-forms.m:[[@LINE-1]]:{{[0-9]+}}: error: a parallel do loop is counted by its for, so it cannot have count
ended(Xs, S) :-
    %$omp parallel do reduction(+)
    ( for(I, 1, 3), fromto(Xs, [X | T], T, []), fromto(0, S0, S1, S) do S1 = S0 + I * X ).
% CHECK: reject-forms.m:[[@LINE-1]]:{{[0-9]+}}: error: a parallel do loop is counted by its for, so it cannot have a fromto that ends the loop
unnamed(N, S) :-
    %$omp parallel do
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-forms.m:[[@LINE-1]]:{{[0-9]+}}: error: a parallel do loop can carry a fromto's value from one iteration to the next only if it is named in a reduction clause
state(N, S) :-
    some [!S] (
        !:S = 0,
        %$omp parallel do
        ( for(I, 1, N) do !:S = !.S + I ),
        S = !.S
    ).
% CHECK: reject-forms.m:[[@LINE-4]]:{{[0-9]+}}: error: a parallel do loop can carry a state variable from one iteration to the next only if it is named in a reduction clause
printing(N, !IO) :-
    %$omp parallel do
    ( for(I, 1, N) do io.write_int(I, !IO) ).
% CHECK: reject-forms.m:[[@LINE-2]]:{{[0-9]+}}: error: a parallel do loop must not thread the io state, since its iterations run in any order
failing(N, S) :-
    %$omp parallel do reduction(+)
    ( for(I, 1, N), fromto(0, S0, S1, S) do I > 0, S1 = S0 + I ).
% CHECK: reject-forms.m:[[@LINE-2]]:{{[0-9]+}}: error: the body of a parallel do loop must be det, but it is semidet
