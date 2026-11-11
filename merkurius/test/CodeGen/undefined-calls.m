% A call of a predicate or function that its module declares but does not
% define, by a clause or a C foreign_proc, is not implemented yet: an error
% at the call, at compile time, rather than an undefined symbol when the
% program links. Inputs/declared_only.m defines the others in each way that
% counts, and calling them is no error. One error is reported for each
% procedure, so each call is in a procedure of its own.
%
% RUN: not merkc -fc1 -emit-merk -I %S/Inputs %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module undefined_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module declared_only, int, list.

main(!IO) :-
    calls_pred(!IO),
    calls_func(!IO),
    closure(!IO),
    defined(!IO).

:- pred calls_pred(io::di, io::uo) is det.
calls_pred(!IO) :-
    undefined_pred(1, !IO).
% CHECK: undefined-calls.m:[[@LINE-1]]:5: error: not yet implemented: code generation for calls of declared_only.undefined_pred/3, which its module declares but does not define yet

:- pred calls_func(io::di, io::uo) is det.
calls_func(!IO) :-
    io.write_int(undefined_func(2), !IO).
% CHECK: undefined-calls.m:[[@LINE-1]]:18: error: not yet implemented: code generation for calls of declared_only.undefined_func/1, which its module declares but does not define yet

:- pred closure(io::di, io::uo) is det.
closure(!IO) :-
    P = undefined_pred,
% CHECK: undefined-calls.m:[[@LINE-1]]:9: error: not yet implemented: code generation for calls of declared_only.undefined_pred/3, which its module declares but does not define yet
    P(3, !IO).

:- pred defined(io::di, io::uo) is det.
defined(!IO) :-
    defined_by_state(defined_func(defined_foreign(4)), !IO),
    defined_by_dcg([1], _).
