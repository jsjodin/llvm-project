% With -fstub-unimplemented, which the standard library's build passes, a
% procedure whose body code generation cannot make yet is not an error: its
% body reports what is not implemented when it runs, and the program exits
% with status 1. A procedure without a signature code generation can make
% is left out, as no call of it can be generated either. Temporary (chunk 1
% step 6, docs/INPROGRESS.md).
%
% RUN: merkc -fc1 -fstub-unimplemented -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fstub-unimplemented %s -o %t
% RUN: not %t 2> %t.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.err

:- module stub_unimplemented.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    io.write_string("before\n", !IO),
    ( if q(3) then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ).

:- pred q(int::in) is semidet.
q(X) :-
    pow(X, 2) > 3.

% OUTPUT:before
% ERROR:Not yet implemented: stub_unimplemented.q_1_0: code generation for calls of int.pow/2

% MERK: merk.proc private @stub_unimplemented.q_1_0(%arg0: i64) attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: %[[WHAT:.*]] = merk.constant "stub_unimplemented.q_1_0: code generation for calls of int.pow/2" : !merk.string
% MERK-NEXT: merk.foreign_call @merk_not_implemented(%[[WHAT]]) : (!merk.string) -> ()
% MERK-NEXT: merk.fail
