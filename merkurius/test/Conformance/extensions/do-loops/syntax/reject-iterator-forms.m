% DESIGN: Loops / Parsing
% The converter checks what needs no types: each iterator is one of the
% forms, a loop has at most one for, the counter of a for or a count is a
% variable, and a for's step is a nonzero int literal. param and
% foreacharg are left out, each with an error that says why.
% ORACLE: none
%
% RUN: not merkc -fc1 -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
:- module reject_iterator_forms.
:- interface.
:- import_module list.
:- pred p1(int::out) is det.
:- pred p2(int::out) is det.
:- pred p3(int::out) is det.
:- pred p4(list(int)::in, int::out) is det.
:- pred p5(int::out) is det.
:- pred p6(list(int)::in, int::out) is det.
:- pred p7(int::out) is det.
:- implementation.
:- import_module int.
p1(S) :- ( for(I, 1, 3), for(J, 1, 3), fromto(0, S0, S1, S) do S1 = S0 + I + J ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:26: error: a loop has at most one for iterator; count(I, Lo, Hi) counts in lockstep with it
p2(S) :- ( for(I, 1, 3, 0), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:25: error: '0' is not a nonzero integer literal: the step of a for iterator is one, so that its direction is known
p3(S) :- N = 2, ( for(I, 1, 3, N), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:32: error: 'N' is not a nonzero integer literal: the step of a for iterator is one, so that its direction is known
p4(Xs, S) :- ( foreach(X, Xs), param(Y), fromto(0, S0, S1, S) do S1 = S0 + X + Y ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:32: error: 'param(Y)': loops have no param iterator; a variable that occurs outside a loop's body is shared by its iterations
p5(S) :- ( foreacharg(A, f(1, 2)), fromto(0, S0, S1, S) do S1 = S0 + A ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:12: error: 'foreacharg(A, f(1, 2))': loops have no foreacharg iterator, since a term's arguments may have different types
p6(Xs, S) :- ( foreach(X, Xs), while(X > 0), fromto(0, S0, S1, S) do S1 = S0 + X ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:32: error: 'while(X > 0)' is not an iterator: expected for(I, Lo, Hi), for(I, Lo, Hi, Step), foreach(X, List), count(I, Lo, Hi) or fromto(First, In, Out, Last)
p7(S) :- ( for(f(I), 1, 3), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-iterator-forms.m:[[@LINE-1]]:16: error: 'f(I)' is not a variable: the first argument of for and count is the loop's counter
