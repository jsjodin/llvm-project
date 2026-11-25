% Malformed pragmas: the wrong number of arguments, an argument of the
% wrong form, an attribute that is not one or is given twice where it may
% not be, and pragmas that are not recognized. mmc reports a syntax error on
% each of these lines too, except the repeated variable of a foreign_proc's
% head, which it reports after parsing, and the pragmas merkc does not yet
% implement (require_tail_recursion, mode_check_clauses and oisu).
%
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module pragma_errors.
:- implementation.
:- pred p(int::in, int::out) is det.
:- type t ---> a ; b.
:- pragma inline(p/2, p/2).
% CHECK: pragma-errors.m:[[@LINE-1]]:11: error: a ':- pragma inline' declaration must have one argument
:- pragma inline.
% CHECK: pragma-errors.m:[[@LINE-1]]:11: error: a ':- pragma inline' declaration must have one argument
:- pragma inline(p).
% CHECK: pragma-errors.m:[[@LINE-1]]:18: error: expected 'Name/Arity', 'pred(Name/Arity)' or 'func(Name/Arity)', but found 'p'
:- pragma obsolete(p(in, out)).
% CHECK: pragma-errors.m:[[@LINE-1]]:20: error: expected 'Name/Arity', 'pred(Name/Arity)' or 'func(Name/Arity)', but found 'p(in, out)'
:- pragma obsolete_proc(p/2).
% CHECK: pragma-errors.m:[[@LINE-1]]:25: error: expected a predicate or function mode such as 'p(in, out)', but found 'p/2'
:- pragma type_spec(p/2, int).
% CHECK: pragma-errors.m:[[@LINE-1]]:26: error: expected a type substitution 'T = Type', but found 'int'
:- pragma memo(p/2, allow_reset).
% CHECK: pragma-errors.m:[[@LINE-1]]:21: error: expected a list of tabling attributes, but found 'allow_reset'
% mmc's hidden argument forms of specified, which it accepts:
:- pragma memo(p/2, [specified([value, output], hidden_arg_value)]).
:- pragma memo(p/2, [specified([value, output], hidden_arg_addr)]).
:- pragma memo(p/2, [specified([value, output], other)]).
% CHECK: pragma-errors.m:[[@LINE-1]]:49: error: expected 'hidden_arg_value' or 'hidden_arg_addr', but found 'other'
:- pragma memo(p/2, [size_limit(x)]).
% CHECK: pragma-errors.m:[[@LINE-1]]:33: error: expected a size limit, but found 'x'
:- pragma memo(p/2, [nosuch]).
% CHECK: pragma-errors.m:[[@LINE-1]]:22: error: expected a tabling attribute: 'allow_reset', 'statistics', 'fast_loose', 'specified(...)', 'size_limit(N)' or 'disable_warning_if_ignored', but found 'nosuch'
:- pragma memo(p/2, [statistics, statistics]).
% CHECK: pragma-errors.m:[[@LINE-1]]:34: error: the attribute 'statistics' is given more than once
:- pragma foreign_export_enum("C", t/0, uppercase).
% CHECK: pragma-errors.m:[[@LINE-1]]:41: error: expected a list of attributes, but found 'uppercase'
:- pragma foreign_export_enum("C", t/0, [nosuch]).
% CHECK: pragma-errors.m:[[@LINE-1]]:42: error: expected 'prefix("Prefix")' or 'uppercase', but found 'nosuch'
:- pragma require_tail_recursion(p/2).
% CHECK: pragma-errors.m:[[@LINE-1]]:4: error: not yet implemented: the ':- pragma require_tail_recursion' declaration
:- pragma mode_check_clauses(p/2).
% CHECK: pragma-errors.m:[[@LINE-1]]:4: error: not yet implemented: the ':- pragma mode_check_clauses' declaration
:- pragma nosuch(p/2).
% CHECK: pragma-errors.m:[[@LINE-1]]:11: error: 'nosuch' is not a recognized pragma
:- pragma 42.
% CHECK: pragma-errors.m:[[@LINE-1]]:11: error: expected a pragma, but found '42'
:- pragma inline(p/2) where equality is p.
% CHECK: pragma-errors.m:[[@LINE-1]]:4: error: only a foreign_type pragma may have 'where' attributes
:- pragma foreign_decl("C", other, "int x;").
% CHECK: pragma-errors.m:[[@LINE-1]]:29: error: expected 'local' or 'exported', but found 'other'
:- pragma foreign_proc("C", 42, [], "").
% CHECK: pragma-errors.m:[[@LINE-1]]:29: error: expected a predicate or function head, but found '42'
:- pragma foreign_proc("C", p(X::in, X::out), [promise_pure], "X = X;").
% CHECK: pragma-errors.m:[[@LINE-1]]:29: error: the variable X appears more than once
:- pragma foreign_enum("C", t/0, [a - 1, b - "2"]).
% CHECK: pragma-errors.m:[[@LINE-1]]:35: error: expected 'Constructor - "Value"', but found 'a - 1'
:- pragma foreign_export("C", p(in, out), "").
% CHECK: pragma-errors.m:[[@LINE-1]]:43: error: expected the foreign name, a non-empty string, but found '""'
:- pragma fact_table(p/2, ages).
% CHECK: pragma-errors.m:[[@LINE-1]]:27: error: expected a file name, but found 'ages'
:- pragma source_file(1).
% CHECK: pragma-errors.m:[[@LINE-1]]:23: error: expected a file name string, but found '1'
:- pragma foreign_code("C").
% CHECK: pragma-errors.m:[[@LINE-1]]:11: error: a ':- pragma foreign_code' declaration must have two arguments
:- pragma foreign_type("C", t, 1).
% CHECK: pragma-errors.m:[[@LINE-1]]:32: error: expected a foreign type descriptor string, but found '1'
:- pragma foreign_import_module("C", "x").
% CHECK: pragma-errors.m:[[@LINE-1]]:38: error: expected a module name, but found '"x"'
:- pragma oisu(p/2).
% CHECK: pragma-errors.m:[[@LINE-1]]:4: error: not yet implemented: the ':- pragma oisu' declaration
:- pragma foreign_export_enum("C", t/0, [prefix("A_"), prefix("B_")]).
% CHECK: pragma-errors.m:[[@LINE-1]]:56: error: the attribute 'prefix' is given more than once
