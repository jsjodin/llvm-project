% A procedure whose clauses use what mode analysis does not check yet has
% no body to generate. Code generation reports it, once for all of a
% predicate's modes, rather than leaving a declaration its callers fail to
% link against, even when nothing calls it. With -fstub-unimplemented, such
% a procedure is a stub that reports this when it runs, a nondet one too.
% A trace goal's state parameters need mutables, whose declarations
% desugaring does not take yet, so these name a mutable not declared.
%
% RUN: not merkc -fc1 -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fc1 -fstub-unimplemented -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=STUB

:- module unchecked_procedures.
:- interface.
:- import_module io, list.
:- pred main(io::di, io::uo) is det.
:- pred traced(int::in, int::out) is det.
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with trace goals with state parameters, which mode analysis does not check yet
:- pred parallel(int::in, int::out, int::out) is det.
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with parallel conjunctions, which mode analysis does not check yet
:- pred member_of(int, list(int)).
% CHECK: unchecked-procedures.m:[[@LINE-1]]:9: error: not yet implemented: code generation for procedures with trace goals with state parameters, which mode analysis does not check yet
:- mode member_of(in, in) is semidet.
:- mode member_of(out, in) is nondet.
:- implementation.
:- import_module int.

main(!IO) :-
    io.write_string("unreached\n", !IO).

traced(X, Y) :-
    trace [state(count, !C)] (!:C = !.C + X),
    Y = X.

parallel(X, Y, Z) :-
    ( Y = X + 1 & Z = X + 2 ).

member_of(X, L) :-
    L = [H | T],
    trace [state(count, !C)] (!:C = !.C + H),
    ( X = H ; member_of(X, T) ).

% STUB: merk.proc @unchecked_procedures.traced_2_0(%arg0: i64) -> i64
% STUB-NEXT: %[[WHAT:.*]] = merk.constant "unchecked_procedures.traced_2_0: code generation for procedures with trace goals with state parameters, which mode analysis does not check yet" : !merk.string
% STUB-NEXT: merk.foreign_call @merk_not_implemented(%[[WHAT]]) : (!merk.string) -> ()
% STUB-NEXT: merk.fail
% STUB: merk.proc @unchecked_procedures.member_of_2_0(
% STUB: merk.proc @unchecked_procedures.member_of_2_1(%arg0: !merk.term<"list.list(int)">) -> i64 attributes {determinism = #merk.determinism<nondet>} {
% STUB-NEXT: %[[WHAT:.*]] = merk.constant "unchecked_procedures.member_of_2_1: code generation for procedures with trace goals with state parameters, which mode analysis does not check yet" : !merk.string
% STUB-NEXT: merk.foreign_call @merk_not_implemented(%[[WHAT]]) : (!merk.string) -> ()
% STUB-NEXT: merk.fail
