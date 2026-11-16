% An instance's method is at least as pure as its declaration in the type
% class (reference manual, "Declaring impure functions and predicates"),
% whether defined by clauses or by name; for one by name, its call is
% unannotated too, as mmc reports. Calls of impure methods are annotated.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module methods.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, string.

main(0).
:- impure pred ip(int::in) is det.
:- pragma foreign_proc("C", ip(X::in), [will_not_call_mercury], "(void) X;").
:- semipure pred sp(int::out) is det.
:- pragma foreign_proc("C", sp(X::out), [will_not_call_mercury, promise_semipure], "X = 0;").
:- impure func ifn(int) = int.
:- pragma foreign_proc("C", ifn(X::in) = (Y::out), [will_not_call_mercury], "Y = X;").
:- pred pp(int::in) is det.
pp(_).
:- semipure pred ssp(string::out) is det.
:- pragma foreign_proc("C", ssp(X::out), [will_not_call_mercury, promise_semipure], "X = (MR_String) \"\";").

:- typeclass c(T) where [impure pred im(T::in) is det].
:- instance c(int) where [pred(im/1) is ip].
:- instance c(string) where [(im(_) :- pp(1))].
:- typeclass d(T) where [pred pm(T::in) is det].
:- instance d(int) where [pred(pm/1) is ip].
% CHECK: [[@LINE-1]]:27: error: the implementation of pred methods.pm/1 in instance methods.d(int) is impure, but the method is declared pure
% CHECK: [[@LINE-2]]:41: error: the call of impure pred methods.ip/1 must be annotated `impure'
:- instance d(string) where [(pm(_) :- impure ip(1))].
% CHECK: [[@LINE-1]]:31: error: the implementation of pred methods.pm/1 in instance methods.d(string) is impure, but the method is declared pure
:- typeclass e(T) where [semipure pred sm(T::out) is det].
:- instance e(int) where [pred(sm/1) is sp].
:- instance e(string) where [(sm(X) :- impure ip(1), semipure ssp(X))].
% CHECK: [[@LINE-1]]:31: error: the implementation of pred methods.sm/1 in instance methods.e(string) is impure, but the method is declared semipure

:- impure pred calls_method is det.
calls_method :- impure im(1).
:- pred calls_method_unannotated is det.
% CHECK: [[@LINE-1]]:9: error: pred methods.calls_method_unannotated/0 is impure, so it must be declared `impure' or promised pure
calls_method_unannotated :- im(1).
% CHECK: [[@LINE-1]]:29: error: the call of impure pred methods.im/1 must be annotated `impure'
