% MANUAL: Typeclass declarations
% Only pred, func and mode declarations may be in a typeclass
% declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_type_in_class.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = T,
    type t ---> t                                   % ERROR
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
