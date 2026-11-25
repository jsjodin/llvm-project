:- module vis_family.
:- interface.
:- import_module io.
:- include_module mid.
:- type counter.
:- func start = counter.
:- func count(counter) = int.
:- pred run(io::di, io::uo) is det.
:- implementation.
:- import_module list.
:- import_module vis_family.mid, vis_family.mid.leaf.

:- type counter ---> counter(int).
:- type secret ---> secret(int).

start = counter(0).
count(counter(N)) = N.

:- func reveal(secret) = int.
reveal(secret(N)) = N.

run(!IO) :-
    io.write_line(peek, !IO),
    io.write_line(count(bump(start)), !IO),
    io.write_line(reveal(wrapped), !IO).
