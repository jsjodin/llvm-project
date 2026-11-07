% A predicate or function of the module needs clauses or a foreign_proc
% for C, as in mmc ("Error: no clauses for predicate `p'/1."). A type
% class's methods are defined by its instances. With -fstub-unimplemented,
% which builds the library while some of its predicates are built into
% code generation or not written yet, none is reported.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fc1 -fdebug-dump-modes -fstub-unimplemented %s 2>&1 | FileCheck %s --check-prefix=STUB --implicit-check-not=error:
% STUB: pred no_clauses.missing/1 mode 0: missing(out) is det

:- module no_clauses.
:- interface.
:- import_module int.

% CHECK: :[[@LINE+1]]:9: error: pred no_clauses.missing/1 has no clauses
:- pred missing(int::out) is det.
% CHECK: :[[@LINE+1]]:9: error: func no_clauses.gone/1 has no clauses
:- func gone(int) = int.

:- pred foreign(int::out) is det.

:- typeclass c(T) where [
    pred method(T::in) is semidet
].

:- implementation.

% CHECK: :[[@LINE+1]]:9: error: pred no_clauses.local/0 has no clauses
:- pred local is semidet.

:- pragma foreign_proc("C", foreign(X::out),
    [will_not_call_mercury, promise_pure], "X = 1;").
