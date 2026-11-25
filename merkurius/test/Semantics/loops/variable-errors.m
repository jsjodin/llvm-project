% The errors for a loop's variables (design document, "Loops", Variables
% and Errors), which name the loop by its line and the user's variables,
% never the translation's predicates:
% - a shared variable, one that occurs around the loop, free at the loop
%   must be bound before it, whether the body binds it or not;
% - a local variable unbound in the body that has the name of another
%   loop's variable says it is local to each iteration of that loop;
% - an error in a loop's body is reported where it is, in the loop, also
%   in a loop nested in another.
% A shared variable that a later goal binds is bound before the loop by
% reordering, and an iterator's outer arguments, foreach's list, count's
% Hi and fromto's Last, may be free (ok.m). A mode error in a declared
% procedure ends mode inference, as in mmc, so each error that inference
% finds has a module of its own.
%
% RUN: split-file %s %t
% RUN: not merkc -fc1 -fdo-loops -emit-merk %t/shared.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHARED --implicit-check-not=error:
% RUN: not merkc -fc1 -fdo-loops -emit-merk %t/local.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOCAL --implicit-check-not=error:
% RUN: not merkc -fc1 -fdo-loops -emit-merk %t/nested.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=NESTED --implicit-check-not=error:
% RUN: not merkc -fc1 -fdo-loops -emit-merk %t/plain.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAIN --implicit-check-not=error:
% RUN: merkc -fc1 -fdo-loops -emit-merk %t/ok.m -o /dev/null 2>&1 | count 0
%
% SHARED: shared.m:7:27: error: in the clause for head_shared(in, out): Sum is shared by the iterations of the loop at line 7, since it occurs around it, so it must be bound before the loop; to give a value out of a loop, use fromto(First, In, Out, Sum) or a state variable
% SHARED: shared.m:9:7: error: in the clause for after_shared(in, out): S is shared by the iterations of the loop at line 9, since it occurs around it, so it must be bound before the loop; to give a value out of a loop, use fromto(First, In, Out, S) or a state variable
% LOCAL: local.m:9:54: error: in the clause for local(in, in, out): in the loop at line 9: S is local to each iteration of the loop at line 8; to give a value out of a loop, use fromto(First, In, Out, S) or a state variable
% NESTED: nested.m:9:58: error: in the clause for nested(in, out): in the loop at line 8: in the loop at line 9: mode error in the call to func int.+/2: the arguments V_9, Z and the result B1 have insts unique, free, free, which match none of its modes
% PLAIN: plain.m:8:54: error: in the clause for plain(in, out): in the loop at line 8: mode error in the call to func int.+/2: the arguments V_6, Z and the result A1 have insts unique, free, free, which match none of its modes

%--- shared.m
:- module shared.
:- interface.
:- import_module list.
:- pred head_shared(list(int)::in, int::out) is det.
:- pred after_shared(list(int)::in, int::out) is det.
:- implementation.
head_shared(Xs, Sum) :- ( foreach(X, Xs) do Sum = X ).
after_shared(Xs, Sum) :-
    ( foreach(X, Xs) do S = X ),
    Sum = S.
%--- local.m
:- module local.
:- interface.
:- import_module list.
:- pred local(list(int)::in, list(int)::in, int::out) is det.
:- implementation.
:- import_module int.
local(Xs, Ys, Sum) :-
    ( foreach(X, Xs) do S = X ),
    ( foreach(Y, Ys), fromto(0, A0, A1, Sum) do A1 = A0 + Y + S ).
%--- nested.m
:- module nested.
:- interface.
:- import_module list.
:- pred nested(list(list(int))::in, int::out) is det.
:- implementation.
:- import_module int.
nested(Xss, Sum) :-
    ( foreach(Xs, Xss), fromto(0, A0, A1, Sum) do
        ( foreach(X, Xs), fromto(A0, B0, B1, A1) do B1 = B0 + X + Z )
    ).
%--- plain.m
:- module plain.
:- interface.
:- import_module list.
:- pred plain(list(int)::in, int::out) is det.
:- implementation.
:- import_module int.
plain(Xs, Sum) :-
    ( foreach(X, Xs), fromto(0, A0, A1, Sum) do A1 = A0 + X + Z ).
%--- ok.m
:- module ok.
:- interface.
:- import_module list.
:- pred reordered(list(int)::in, list(int)::out) is det.
:- pred outer_free(list(int)::in, list(int)::out, int::out, int::out) is det.
:- implementation.
:- import_module int.
reordered(Xs, Ys) :-
    ( foreach(X, Xs), foreach(Y, Ys) do Y = X + S ),
    S = 3.
outer_free(Xs, Ys, N, Last) :-
    ( foreach(X, Xs), foreach(Y, Ys), count(_, 1, N), fromto(0, _, X, Last) do
        Y = X * 2
    ).
