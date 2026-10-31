% A foreign_proc is impure unless its promise_pure or promise_semipure
% attribute, or a pragma, says otherwise (reference manual, "Foreign code
% attributes"), and its purity must be the declared one; a pragma stands
% for both. Only those for C count, as mmc compiles to C.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module foreign.
:- interface.
:- pred main(int::out) is det.
:- implementation.

main(0).
:- pred promised(int::out) is det.
:- pragma foreign_proc("C", promised(X::out), [will_not_call_mercury, promise_pure], "X = 0;").
:- semipure pred promised_semipure(int::out) is det.
:- pragma foreign_proc("C", promised_semipure(X::out), [will_not_call_mercury, promise_semipure], "X = 0;").
:- impure pred unpromised(int::out) is det.
:- pragma foreign_proc("C", unpromised(X::out), [will_not_call_mercury], "X = 0;").
:- pred by_pragma(int::out) is det.
:- pragma promise_pure(by_pragma/1).
:- pragma foreign_proc("C", by_pragma(X::out), [will_not_call_mercury], "X = 0;").
:- pred with_clause(int::out) is det.
:- pragma foreign_proc("C", with_clause(X::out), [will_not_call_mercury, promise_pure], "X = 0;").
with_clause(0).
:- pred other_language(int::out) is det.
:- pragma foreign_proc("C", other_language(X::out), [will_not_call_mercury, promise_pure], "X = 0;").
:- pragma foreign_proc("Java", other_language(X::out), [will_not_call_mercury], "X = 0;").

:- pred pure_unpromised(int::out) is det.
% CHECK: [[@LINE-1]]:9: error: pred foreign.pure_unpromised/1 is impure, so it must be declared `impure' or promised pure
:- pragma foreign_proc("C", pure_unpromised(X::out), [will_not_call_mercury], "X = 0;").
% CHECK: [[@LINE-1]]:4: error: this foreign_proc for pred foreign.pure_unpromised/1 is impure, but pred foreign.pure_unpromised/1 is declared pure
:- pred pure_semipure(int::out) is det.
% CHECK: [[@LINE-1]]:9: error: pred foreign.pure_semipure/1 is semipure, so it must be declared `semipure' or promised pure
:- pragma foreign_proc("C", pure_semipure(X::out), [will_not_call_mercury, promise_semipure], "X = 0;").
% CHECK: [[@LINE-1]]:4: error: this foreign_proc for pred foreign.pure_semipure/1 is semipure, but pred foreign.pure_semipure/1 is declared pure
:- semipure pred semipure_unpromised(int::out) is det.
% CHECK: [[@LINE-1]]:18: error: pred foreign.semipure_unpromised/1 is impure, so it must be declared `impure' or promised semipure
:- pragma foreign_proc("C", semipure_unpromised(X::out), [will_not_call_mercury], "X = 0;").
% CHECK: [[@LINE-1]]:4: error: this foreign_proc for pred foreign.semipure_unpromised/1 is impure, but pred foreign.semipure_unpromised/1 is declared semipure
:- impure pred impure_promised(int::out) is det.
:- pragma foreign_proc("C", impure_promised(X::out), [will_not_call_mercury, promise_pure], "X = 0;").
% CHECK: [[@LINE-1]]:4: error: this foreign_proc for pred foreign.impure_promised/1 is pure, but pred foreign.impure_promised/1 is declared impure
:- func function = int.
% CHECK: [[@LINE-1]]:9: error: func foreign.function/0 is impure, so it must be declared `impure' or promised pure
:- pragma foreign_proc("C", function = (X::out), [will_not_call_mercury], "X = 0;").
% CHECK: [[@LINE-1]]:4: error: this foreign_proc for func foreign.function/0 is impure, but func foreign.function/0 is declared pure
