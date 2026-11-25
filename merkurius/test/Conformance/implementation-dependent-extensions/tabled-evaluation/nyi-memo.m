% MANUAL: Tabled evaluation
% A memo pragma, which makes a doubly recursive function linear.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: tabled evaluation (`:- pragma memo')

:- module nyi_memo.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- func fib(int) = int.
:- pragma memo(fib/1).
fib(N) = ( if N < 2 then N else fib(N - 1) + fib(N - 2) ).

main(!IO) :- io.write_int(fib(60), !IO), io.nl(!IO).
