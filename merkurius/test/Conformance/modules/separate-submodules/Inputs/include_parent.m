:- module include_parent.
:- interface.
:- import_module io.
:- include_module shapes.
:- pred run(io::di, io::uo) is det.
:- implementation.
:- include_module helpers.
:- import_module list.
:- import_module include_parent.shapes, include_parent.helpers.

:- func scale = int.
scale = 3.

run(!IO) :-
    io.write_line(total([square(2), circle(1)]), !IO).
