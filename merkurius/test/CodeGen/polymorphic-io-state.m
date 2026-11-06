% A polymorphic procedure whose type variable is the state of the world, as
% when the state is threaded through a polymorphic accumulator. io.state is
% an abstract type that io does not define, so its descriptor is a dummy's,
% linkonce in each module that uses it, with no functors. The output is
% mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module polymorphic_io_state.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred thread(pred(A, A), A, A).
:- mode thread(pred(di, uo) is det, di, uo) is det.
:- pragma no_inline(thread/3).
thread(P, !A) :-
    P(!A).

main(!IO) :-
    thread((pred(!.S::di, !:S::uo) is det :- io.write_string("hello\n", !S)),
        !IO),
    thread(io.write_string("again\n"), !IO).

% OUTPUT:hello
% OUTPUT-NEXT:again

% MERK: merk.type_ctor "io.state/0" module "io" name "state" kind "dummy" linkage "linkonce" arity 0 unify @"unify_entry.io.state/0" compare @"compare_entry.io.state/0" functors []
