% Each mode of a type class method declares its determinism (reference
% manual, "Typeclass declarations"), in a separate mode declaration or in
% the method's predicate or function declaration, though the class is in
% the implementation section. A function method without modes has the
% default det mode.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module method_determinism.
:- interface.
:- implementation.

:- typeclass c(T) where [
    pred p(T, int),
    mode p(in, out),
% CHECK: [[@LINE-1]]:10: error: the method pred method_determinism.p/2 has no determinism declaration in this mode declaration
    pred q(T::in, int::out),
% CHECK: [[@LINE-1]]:10: error: the method pred method_determinism.q/2 has no determinism declaration
    pred r(T::in, int::out) is det,
    func f(T) = int
].
