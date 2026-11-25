% DESIGN: Loops / Under a parallel directive
% A Mercury variable has one value, so a private clause naming a variable
% that the iterations share, one that occurs around the loop, is an error,
% and so is one naming a value the loop carries from one iteration to the
% next, or a variable that is not in the loop's body, as one only in its
% bounds.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopenmp -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_private.
:- interface.
:- import_module array.
:- pred shared(int::in, int::in, int::out) is det.
:- pred carried(int::in, int::out) is det.
:- pred state(int::in, int::out) is det.
:- pred absent(int::in, int::out) is det.
:- pred bound(int::in, int::out) is det.
:- pred term(int::in, int::out) is det.
:- pred scale(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- implementation.
:- import_module int.

shared(K, N, S) :-
    %$omp parallel do private(K) reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + K * I ).
% CHECK: reject-private.m:[[@LINE-2]]:{{[0-9]+}}: error: K is shared by the iterations of the loop, since it occurs around it, and has one value, so it cannot be private
carried(N, S) :-
    %$omp parallel do private(S) reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-private.m:[[@LINE-2]]:{{[0-9]+}}: error: S is carried from one iteration of the loop to the next, so it cannot be private
state(N, S) :-
    some [!S] (
        !:S = 0,
        %$omp parallel do private(S) reduction((+) : S)
        ( for(I, 1, N) do !:S = !.S + I ),
        S = !.S
    ).
% CHECK: reject-private.m:[[@LINE-4]]:{{[0-9]+}}: error: S is carried from one iteration of the loop to the next, so it cannot be private
absent(N, S) :-
    %$omp parallel do private(Z) reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-private.m:[[@LINE-2]]:{{[0-9]+}}: error: Z does not occur in the loop's body
bound(K, S) :-
    %$omp parallel do private(K) reduction((+) : S)
    ( for(I, 1, K), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-private.m:[[@LINE-2]]:{{[0-9]+}}: error: K does not occur in the loop's body
term(N, S) :-
    %$omp parallel do private(f(X)) reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do X = I, S1 = S0 + X ).
% CHECK: reject-private.m:[[@LINE-2]]:{{[0-9]+}}: error: a private clause names variables
scale(K, !A) :-
    %$omp parallel do private(K)
    ( for(I, 0, 9) do
        !A ^ unsafe_elem(I) := K * I
    ).
% CHECK: reject-private.m:[[@LINE-4]]:{{[0-9]+}}: error: K is shared by the iterations of the loop, since it occurs around it, and has one value, so it cannot be private
