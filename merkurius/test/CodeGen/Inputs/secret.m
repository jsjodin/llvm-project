:- module secret.
:- interface.
:- import_module io.

:- type box(T).
:- type colour.
:- type pair_of(T).

:- func box(T) = box(T).
:- func unbox(box(T)) = T.
:- func red = colour.
:- func blue = colour.
:- func name(colour) = string.
:- func pair_of(T, T) = pair_of(T).
:- pred swap(pair_of(T)::in, pair_of(T)::out) is det.
:- pred show_pair(pair_of(T)::in, io::di, io::uo) is det.

:- implementation.

:- type box(T) ---> box(T, int).
:- type colour ---> crimson ; navy ; green.
:- type pair_of(T) ---> two(T, T).

box(X) = box(X, 0).
unbox(box(X, _)) = X.
red = crimson.
blue = navy.
name(crimson) = "red".
name(navy) = "blue".
name(green) = "green".
pair_of(A, B) = two(A, B).
swap(two(A, B), two(B, A)).
show_pair(P, !IO) :-
    io.write(P, !IO),
    io.nl(!IO).
