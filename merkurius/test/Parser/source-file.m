% A source_file pragma names the file that the lines after it come from
% (reference manual, "Source file name"): messages give it, with the line
% numbers counting on from the pragma until a '#' line number directive
% sets them, and a later pragma names another file in turn.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module source_file.
:- interface.
:- pred p(int::out) is det.
:- pred q(int::out) is det.
:- pred r(int::out) is det.
:- implementation.
:- pragma source_file("generator.m.in").
p(X) :- X = "a".
% CHECK: {{^}}generator.m.in:15:13: error: type error in unification of variable X and `"a"'
#20
q(X) :- X = "b".
% CHECK: {{^}}generator.m.in:20:13: error: type error in unification of variable X and `"b"'
:- pragma source_file("source-file.m").
#40
r(X) :- X = "c".
% CHECK: {{^}}source-file.m:40:13: error: type error in unification of variable X and `"c"'
