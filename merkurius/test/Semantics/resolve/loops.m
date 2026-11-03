% A loop's foreach names the list type, which needs the list module, as
% list syntax does; the names in a loop's body are resolved as anywhere.
%
% RUN: not merkc -fc1 -fdo-loops -fdebug-dump-resolved-names %s 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module loops.
:- interface.
:- pred p(int::out) is det.
:- implementation.
:- import_module int.

p(N) :-
    ( for(I, 1, 3), fromto(0, S0, S1, N) do S1 = S0 + undefined(I) ),
    ( foreach(X, L) do X = 1 ),
    q(L).
% CHECK: loops.m:[[@LINE-3]]:55: error: undefined symbol undefined/1
% CHECK: loops.m:[[@LINE-3]]:7: error: foreach(X, L) needs the list module, which is not imported

:- pred q(T::out) is det.
q(_) :- true.
