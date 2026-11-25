% MANUAL: Nested submodules
% The manual says a nested submodule's end_module declaration is
% mandatory, even when it is the last thing in the file, but mmc accepts
% one without it there (design document, "Where mmc and the manual
% disagree").
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: nested submodules; nyi_no_end_module.child can be a separate submodule, in a file of its own, nyi_no_end_module.child.m, which the implementation section of nyi_no_end_module includes with `:- include_module child.`

:- module nyi_no_end_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO).

    :- module child.
    :- interface.
    :- type t ---> t.
