% A module laid out carelessly, whose canonical layout is messy.canonical.m.

:- module messy.
:- interface.
:- import_module io, list.

:- type shape
    --->    circle(float)
    ;       square(side::float)
    ;       rectangle(width::float, height::float).

:- pred main(io::di, io::uo) is det.
:- func area(shape) = float.
:- pred classify(list(int)::in, string::out) is det.

:- implementation.
:- import_module float, int, string.

main(!IO) :-
    io.write_string("start\n", !IO), % say hello
    Shapes = [circle(1.0), square(2.0), rectangle(float(0x10), 3.0e2)],
    list.foldl(
        (pred(S::in, !.A::in, !:A::out) is det :- !:A = !.A + area(S)),
        Shapes,
        0.0,
        Total),
    /* the total */
    io.write_float(Total, !IO),
    ( if Total > 100.0 then
        io.write_string(" big\n", !IO)
    else if Total > 10.0 then
        io.write_string(" medium\n", !IO)
    else
        io.write_string(" small\n", !IO)
    ),
    classify([1, 2, 3], C),
    io.write_string(C, !IO).

area(circle(R)) = 3.14159 * R * R.
area(square(S)) = S * S.
area(rectangle(W, H)) = W * H.

classify(Xs, Class) :-
    (
        Xs = [],
        Class = "empty"
    ;
        Xs = [_],
        Class = "one"
    ;
        Xs = [_, _ | _],
        Class = string.format("%d elements", [i(list.length(Xs))])
    ).
