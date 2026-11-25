% MANUAL: Tabled evaluation
% A memo pragma with the allow_reset and statistics attributes, and a
% call of the reset predicate that allow_reset defines.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: tabled evaluation (`:- pragma memo')

:- module nyi_allow_reset.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- func fib(int) = int.
:- pragma memo(fib/1, [allow_reset, statistics]).
fib(N) = ( if N < 2 then N else fib(N - 1) + fib(N - 2) ).

main(!IO) :-
    io.write_int(fib(30), !IO), io.nl(!IO),
    table_reset_for_fib_1(!IO),
    io.write_int(fib(20), !IO), io.nl(!IO).
