% Calls of obsolete predicates, functions and procedures are warned of,
% with the replacements their pragmas suggest (reference manual,
% "Obsolescence"): the module's own and an imported module's, by a call or
% a function application, or by a closure made from the name. A call of an
% obsolete procedure in another mode is not, and neither is a call in a
% predicate that is obsolete itself, as in mmc.
%
% RUN: merkc -fc1 -fdebug-dump-determinism -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=warning: %s

:- module obsolete.
:- interface.
:- import_module list.
:- pred main(list(int)::out) is det.
:- implementation.
:- import_module int, obsolete_lib.

:- pragma obsolete(old/1).
:- func old(int) = int.
old(X) = newer(X).
:- pragma obsolete(newer/1).
:- func newer(int) = int.
newer(X) = X + 1.
:- pragma obsolete(old2/2, [new/2, new3/2]).
:- pred old2(int::in, int::out) is det.
old2(X, X).
:- pred new(int::in, int::out) is det.
new(X, X).
:- pred new3(int::in, int::out) is det.
new3(X, X).
:- pred m(int, int).
:- mode m(in, out) is det.
:- mode m(out, in) is det.
:- pragma obsolete_proc(m(out, in), [new/2]).
m(X, X).

main([A, B, C, D, E, F, G, H]) :-
    A = old(1),
% CHECK: [[@LINE-1]]:9: warning: call to obsolete func obsolete.old/1
    old2(A, B),
% CHECK: [[@LINE-1]]:5: warning: call to obsolete pred obsolete.old2/2
% CHECK: the possible suggested replacements are new/2 and new3/2
    m(B, C),
    m(D, C),
% CHECK: [[@LINE-1]]:5: warning: call to obsolete procedure m(out, in) of pred obsolete.m/2
% CHECK: the suggested replacement is new/2
    Old = old,
% CHECK: [[@LINE-1]]:11: warning: call to obsolete func obsolete.old/1
    E = Old(D),
    F = lib_old(E),
% CHECK: [[@LINE-1]]:9: warning: call to obsolete func obsolete_lib.lib_old/1
% CHECK: the suggested replacement is lib_new/1
    lib_m(F, G),
% CHECK: [[@LINE-1]]:5: warning: call to obsolete procedure lib_m(in, out) of pred obsolete_lib.lib_m/2
    lib_m(H, G).
