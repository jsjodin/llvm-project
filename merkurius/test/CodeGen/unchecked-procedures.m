% A procedure whose clauses use what mode analysis does not check yet has
% no body to generate. Code generation reports it, once for all of a
% predicate's modes, rather than leaving a declaration its callers fail to
% link against, even when nothing calls it. With -fstub-unimplemented, a
% procedure with a signature code generation can make is a stub that
% reports this when it runs, and one without, a nondet procedure, is left
% out.
%
% RUN: not merkc -fc1 -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fc1 -fstub-unimplemented -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=STUB

:- module unchecked_procedures.
:- interface.
:- import_module io, list.
:- pred main(io::di, io::uo) is det.
:- pred one_mode(int, int).
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with clauses for one mode, which mode analysis does not check yet
:- mode one_mode(in, out) is det.
:- mode one_mode(out, in) is det.
:- pred traced(int::in, int::out) is det.
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with trace goals, which mode analysis does not check yet
:- pred parallel(int::in, int::out, int::out) is det.
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with parallel conjunctions, which mode analysis does not check yet
:- pred member_of(int, list(int)).
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with clauses for one mode, which mode analysis does not check yet
:- mode member_of(in, in) is semidet.
:- mode member_of(out, in) is nondet.
:- implementation.
:- import_module int.

main(!IO) :-
    io.write_string("unreached\n", !IO).

:- pragma promise_equivalent_clauses(one_mode/2).
one_mode(X::in, Y::out) :-
    Y = X + 1.
one_mode(X::out, Y::in) :-
    X = Y - 1.

traced(X, Y) :-
    trace [io(!IO)] (io.write_int(X, !IO)),
    Y = X.

parallel(X, Y, Z) :-
    ( Y = X + 1 & Z = X + 2 ).

:- pragma promise_equivalent_clauses(member_of/2).
member_of(X::in, L::in) :-
    L = [H | T],
    ( if X = H then true else member_of(X, T) ).
member_of(X::out, L::in) :-
    L = [H | T],
    ( X = H ; member_of(X, T) ).

% STUB: merk.proc @unchecked_procedures.one_mode_2_0(%arg0: i64) -> i64
% STUB-NEXT: %[[WHAT:.*]] = merk.constant "unchecked_procedures.one_mode_2_0: code generation for procedures with clauses for one mode, which mode analysis does not check yet" : !merk.string
% STUB-NEXT: merk.foreign_call @merk_not_implemented(%[[WHAT]]) : (!merk.string) -> ()
% STUB-NEXT: merk.fail
% STUB: merk.proc @unchecked_procedures.member_of_2_0(
% STUB-NOT: member_of_2_1
