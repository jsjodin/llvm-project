% The determinisms of procedures with I/O state arguments, as mmc checks
% them: one with exactly two arguments of type io.state, one bound at the
% start and one free, whatever their uniqueness, must be det, cc_multi or
% erroneous, by its declared determinism, or its inferred one if it has
% none. An instance method's is reported at its definition in the
% instance, as are its other determinism errors. An assignment to a
% variable not used again does not share its value.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module io_state.
:- interface.
:- import_module io.
:- import_module list.

% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for is_multi(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi
:- pred is_multi(io::di, io::uo) is multi.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for is_nondet(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is nondet
:- pred is_nondet(io::di, io::uo) is nondet.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for is_semidet(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is semidet
:- pred is_semidet(io::di, io::uo) is semidet.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for is_failure(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is failure
:- pred is_failure(io::di, io::uo) is failure.
:- pred is_erroneous(io::di, io::uo) is erroneous.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for is_cc_nondet(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is cc_nondet
:- pred is_cc_nondet(io::di, io::uo) is cc_nondet.
:- pred is_cc_multi(io::di, io::uo) is cc_multi.
:- pred is_det(io::di, io::uo) is det.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for mostly(mdi, muo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi
:- pred mostly(io::mdi, io::muo) is multi.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for shared(in, out): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is semidet
:- pred shared(io::in, io::out) is semidet.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for more(in, di, out, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is nondet
:- pred more(int::in, io::di, int::out, io::uo) is nondet.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for function(di) = uo: a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi
:- func function(io::di) = (io::uo) is multi.
:- type my_io == io.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for equivalent(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi
:- pred equivalent(my_io::di, my_io::uo) is multi.
:- pred modes(io, io).
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for modes(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi
:- mode modes(di, uo) is multi.
:- mode modes(in, out) is det.
:- pred one(io::ui) is multi.
:- pred output(io::uo) is multi.
:- pred inputs(io::di, io::di) is multi.
:- pred outputs(io::uo, io::uo) is multi.
:- pred two_pairs(io::di, io::uo, io::di, io::uo) is semidet.
:- pred list(list(io)::di, list(io)::uo) is multi.
:- pred calls(int::out, io::di, io::uo) is det.
:- typeclass tc(T) where [ pred method(T::in, io::di, io::uo) is semidet ].
:- instance tc(int).
:- type t ---> t(int).
:- pred dead_assignment(t::di, t::uo) is det.

:- implementation.
:- import_module int.

:- pred m(int::out) is multi.
m(1).
m(2).

:- instance tc(int) where [
% ERR-DAG: io_state.m:[[@LINE+1]]:5: error: invalid determinism for the method method(in, di, uo) of instance io_state.tc(int): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is semidet
    method(X, !IO) :- X > 1
].

is_multi(!IO).
is_nondet(!IO) :- 1 > 2.
is_semidet(!IO) :- 1 > 2.
is_failure(!IO) :- fail.
is_erroneous(!IO) :- is_erroneous(!IO).
is_cc_nondet(!IO) :- 1 > 2.
is_cc_multi(!IO).
is_det(!IO).
mostly(!IO).
shared(!IO).
more(X, IO0, X, IO) :- IO = IO0.
function(IO) = IO.
equivalent(!IO).
modes(!IO).
one(_).
% ERR-DAG: io_state.m:[[@LINE+1]]:15: error: in the clause for output(uo): mode error: argument 1, IO, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
output(IO) :- output(IO).
inputs(_, _).
% ERR-DAG: io_state.m:[[@LINE+1]]:21: error: in the clause for outputs(uo, uo): mode error: argument 1, IO, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
outputs(IO, IO2) :- outputs(IO, IO2).
two_pairs(!IO, !IO2) :- 1 > 2.
list(!L).
% ERR-DAG: io_state.m:[[@LINE+1]]:41: error: in the clause for calls(out, di, uo): mode error: argument 3, IO!2, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
calls(X, !IO) :- inferred_semidet(!IO), inferred_multi(X, !IO).
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for inferred_semidet(di, uo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is semidet
:- pred inferred_semidet(io::di, io::uo).
inferred_semidet(!IO) :- 1 > 2.
% ERR-DAG: io_state.m:[[@LINE+1]]:9: error: invalid determinism for inferred_multi(out, mdi, muo): a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is multi
:- pred inferred_multi(int::out, io::mdi, io::muo).
inferred_multi(X, !IO) :- m(X).
dead_assignment(T0, T) :- T1 = T0, T = T0.
