:- module gc_scalar_word_spawn.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

main(!IO) :-
    get(1, X, !IO),
    get(2, Y, !IO),
    spawn(gpu(0), a(X, Y), A, !IO),
    spawn(gpu(0), b(A, X, Y), _, !IO).

:- process a(int, int) receives int on gpu.
a(_, _, _Self, !IO).

:- process b(mailbox(int), int, int) receives int on gpu.
b(_, _, _, _Self, !IO).

:- pred get(int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", get(X::in, Y::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure], "Y = X + 1;").
