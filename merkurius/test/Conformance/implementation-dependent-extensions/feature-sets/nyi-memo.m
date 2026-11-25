% MANUAL: Feature sets
% The memo feature, which merkc does not yet have.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: tabled evaluation (the feature 'memo')

:- module nyi_memo.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma require_feature_set([memo]).
main(!IO) :- io.write_string("ok\n", !IO).
