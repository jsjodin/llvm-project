% Pragmas that may not be in the interface section. The reference manual
% says so only of foreign_export_enum; mmc reports each of these as
% "`:- pragma inline' declaration in module interface", and accepts the
% termination pragmas, obsolete, obsolete_proc, type_spec, foreign_type,
% foreign_import_module and source_file there (design document, "Where mmc
% and the manual disagree"). Each pragma also appears where it belongs, in
% the implementation section, without an error.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module interface_pragmas.
:- interface.
:- import_module int.
:- type col ---> red ; blue.
:- type ft.
:- pred p(int::in) is det.
:- pred q(T::in) is det.
:- pred r(int::out) is det.
:- pred s(int::out) is det.
:- pred t(int::in) is det.
:- semipure pred u(int::in) is det.
:- pred f(int::out) is multi.

:- pragma inline(p/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma inline' may not be in the interface section
:- pragma no_inline(t/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma no_inline' may not be in the interface section
:- pragma consider_used(p/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma consider_used' may not be in the interface section
:- pragma no_determinism_warning(p/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma no_determinism_warning' may not be in the interface section
:- pragma promise_pure(p/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma promise_pure' may not be in the interface section
:- pragma promise_semipure(u/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma promise_semipure' may not be in the interface section
:- pragma promise_equivalent_clauses(p/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma promise_equivalent_clauses' may not be in the interface section
:- pragma memo(p/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma memo' may not be in the interface section
:- pragma loop_check(t/1).
% CHECK: [[@LINE-1]]:4: error: `:- pragma loop_check' may not be in the interface section
:- pragma fact_table(f/1, "f.facts").
% CHECK: [[@LINE-1]]:4: error: `:- pragma fact_table' may not be in the interface section
:- pragma require_feature_set([memo]).
% CHECK: [[@LINE-1]]:4: error: `:- pragma require_feature_set' may not be in the interface section
:- pragma foreign_decl("C", "int interface_pragmas_x;").
% CHECK: [[@LINE-1]]:4: error: `:- pragma foreign_decl' may not be in the interface section
:- pragma foreign_code("C", "int interface_pragmas_y;").
% CHECK: [[@LINE-1]]:4: error: `:- pragma foreign_code' may not be in the interface section
:- pragma foreign_proc("C", r(X::out),
    [will_not_call_mercury, promise_pure], "X = 1;").
% CHECK: [[@LINE-2]]:4: error: `:- pragma foreign_proc' may not be in the interface section
:- pragma foreign_enum("C", col/0, [red - "1", blue - "2"]).
% CHECK: [[@LINE-1]]:4: error: `:- pragma foreign_enum' may not be in the interface section
:- pragma foreign_export_enum("C", col/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma foreign_export_enum' may not be in the interface section
:- pragma foreign_export("C", p(in), "interface_pragmas_p").
% CHECK: [[@LINE-1]]:4: error: `:- pragma foreign_export' may not be in the interface section

% Accepted in the interface.
:- pragma terminates(p/1).
:- pragma does_not_terminate(t/1).
:- pragma check_termination(s/1).
:- pragma obsolete(q/1).
:- pragma obsolete_proc(t(in)).
:- pragma type_spec(q/1, T = int).
:- pragma foreign_type("C", ft, "int").
:- pragma foreign_import_module("C", int).

:- implementation.

% Accepted in the implementation.
:- pragma inline(s/1).

p(_).
q(_).
s(1).
t(_).
u(_).
f(1).
