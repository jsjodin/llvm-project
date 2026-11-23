% An imported abstract type that is an equivalence of a type of its
% module's implementation section, which importing modules cannot see, is
% not yet seen through.
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% CHECK: hidden_private.m:12:19: error: not yet implemented: module hidden_private's abstract type secret/0 is an equivalence of private, a type of its implementation section, which importing modules cannot see

:- module hidden_private_client.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module hidden_private.

main(0) :-
    _ = secret(1).
