% A module with obsolete exports, for obsolete.m.
:- module obsolete_lib.
:- interface.
:- pragma obsolete(lib_old/1, [lib_new/1]).
:- func lib_old(int) = int.
:- func lib_new(int) = int.
:- pred lib_m(int, int).
:- mode lib_m(in, out) is det.
:- mode lib_m(out, in) is det.
:- pragma obsolete_proc(lib_m(in, out)).
:- implementation.
lib_old(X) = X.
lib_new(X) = X.
lib_m(X, X).
