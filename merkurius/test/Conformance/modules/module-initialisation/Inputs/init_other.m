:- module init_other.
:- interface.
:- func answer = int.
:- implementation.
:- import_module io.
:- initialise hello/2.
:- pred hello(io::di, io::uo) is det.
hello(!IO) :-
    io.write_string("init_other initialised\n", !IO).
answer = 42.
