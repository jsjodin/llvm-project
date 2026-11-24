% A foreign_decl that is local is not seen by the foreign code of the
% modules that import its module, as one that is not local is (reference
% manual, "Adding foreign declarations"); clang reports the name it would
% have declared.
%
% RUN: not merkc -I %S/../Conformance/foreign-language-interface/adding-foreign-declarations/Inputs \
% RUN:   %s %S/../Conformance/foreign-language-interface/adding-foreign-declarations/Inputs/foreign_decl_shared.m \
% RUN:   -o %t 2>&1 | FileCheck %s

:- module foreign_decl_local.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module foreign_decl_shared.
:- func shared = int.
:- pragma foreign_proc("C", shared = (X::out),
    [promise_pure, will_not_call_mercury], "X = SHARED_ONE + LOCAL_ONE;").
% CHECK: foreign-decl-local.m:[[@LINE-1]]:{{[0-9]+}}: error: use of undeclared identifier 'LOCAL_ONE'
main(!IO) :- io.write_int(shared, !IO), io.nl(!IO).
