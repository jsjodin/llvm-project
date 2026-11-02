% A call whose arguments match more than one mode is of one that does not
% clobber an argument used after it, if there is one, as in mmc: the
% second append, with V1 used after it, is append(in, in, out), and is
% scheduled where it is, not append(di, di, uo), delayed until V1 is no
% longer used.
%
% RUN: merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s

:- module clobber_choice.
:- interface.
:- import_module list.
:- func f(string) = list(string).
:- implementation.
f(_) = O :-
    list.append([], [], V1),
    list.append(["a"], V1, _),
    O = V1.
% CHECK: call pred list.append/3 mode 0: append(V_{{[0-9]+}}, V_{{[0-9]+}}, V1)
% CHECK: call pred list.append/3 mode 1: append(V_{{[0-9]+}}, V1, V_{{[0-9]+}})
% CHECK-NEXT: assign O := V1
