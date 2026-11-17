% A module-qualified name is not an operator term, so the parenthesized
% form leaves it alone, even with several tokens or with an operator as its
% functor, as a.(u > e), whose name's tokens are a.( . Before 2026-11-17
% the name of a bound inst's functor was wrapped, so bound(a.(u > e))
% became bound((a.() u > e)), which reads as another tree. Found by the
% term fuzzer.
%
% RUN: merkc -fc1 -fdebug-unparse-parenthesized-no-sema %s | FileCheck %s
% The parenthesized form reads as the same parse tree as the source.
% RUN: merkc -fc1 -fdebug-unparse-parenthesized-no-sema %s > %t.m
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s > %t.tree
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %t.m | diff %t.tree -
%
% CHECK: {{^}}:- inst operator == bound(a.(u > e)).{{$}}
% CHECK: {{^}}:- inst prefix == bound(a.(-u)).{{$}}
% CHECK: {{^}}:- inst applied == bound(a.f(u) ; a.(u)).{{$}}
% CHECK: {{^}}:- type alias == list.list(int).{{$}}
% CHECK: {{^}}:- mode m == ((a.(u > e)) >> ground).{{$}}

:- module qualified_names.
:- interface.
:- implementation.
:- import_module list.

:- inst operator == bound(a.(u > e)).
:- inst prefix == bound(a.(-u)).
:- inst applied == bound(a.f(u) ; a.(u)).
:- type alias == list.list(int).
:- mode m == a.(u > e) >> ground.
