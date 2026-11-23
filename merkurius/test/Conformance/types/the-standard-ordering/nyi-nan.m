% MANUAL: The standard ordering
% Comparing a NaN throws an exception.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for procedures with try goals, which mode analysis does not check yet

:- module nyi_nan.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, float.

main(!IO) :-
    NaN = float.infinity - float.infinity,
    ( try []
        compare(R, NaN, 1.0)
    then
        io.write_line(R, !IO)
    catch_any E ->
        io.write_string("exception: ", !IO),
        io.write_line(E, !IO)
    ).
