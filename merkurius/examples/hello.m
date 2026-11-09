% Hello, world: a string constant and a call of io.write_string/3, which
% the standard library implements with a foreign_proc for C.

:- module hello.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    io.write_string("Hello, world!\n", !IO).
