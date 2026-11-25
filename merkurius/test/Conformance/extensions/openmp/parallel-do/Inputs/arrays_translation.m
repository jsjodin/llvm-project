% The translation of run-arrays.m's loops, without their directives.
:- module arrays_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

:- pred fill(int::in, int::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo) is det.
fill(Lo, Hi, Step, K, !A) :-
    ( if Step > 0 then
        up(Lo, Hi, K, !A)
    else
        down(Hi, Lo, K, !A)
    ).

:- pred up(int::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo) is det.
up(I, Hi, K, !A) :-
    ( if I > Hi then
        true
    else
        !A ^ unsafe_elem(I) := K * I,
        up(I + 3, Hi, K, !A)
    ).

:- pred down(int::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo) is det.
down(I, Lo, K, !A) :-
    ( if I < Lo then
        true
    else
        ( if I mod 4 = 0 then V = -I else V = I + K ),
        !A ^ unsafe_elem(I) := V,
        down(I - 2, Lo, K, !A)
    ).

:- pred matmul(int::in, array(int)::in, array(int)::in,
    array(int)::array_di, array(int)::array_uo) is det.
matmul(N, A, B, !C) :-
    rows(0, N, A, B, !C).

:- pred rows(int::in, int::in, array(int)::in, array(int)::in,
    array(int)::array_di, array(int)::array_uo) is det.
rows(I, N, A, B, !C) :-
    ( if I > N - 1 then
        true
    else
        columns(I, 0, N, A, B, !C),
        rows(I + 1, N, A, B, !C)
    ).

:- pred columns(int::in, int::in, int::in, array(int)::in, array(int)::in,
    array(int)::array_di, array(int)::array_uo) is det.
columns(I, J, N, A, B, !C) :-
    ( if J > N - 1 then
        true
    else
        dot(I, J, 0, N, A, B, 0, Sum),
        !C ^ unsafe_elem(I * N + J) := Sum,
        columns(I, J + 1, N, A, B, !C)
    ).

:- pred dot(int::in, int::in, int::in, int::in, array(int)::in,
    array(int)::in, int::in, int::out) is det.
dot(I, J, K, N, A, B, S0, Sum) :-
    ( if K > N - 1 then
        Sum = S0
    else
        S1 = S0 + A ^ unsafe_elem(I * N + K) * B ^ unsafe_elem(K * N + J),
        dot(I, J, K + 1, N, A, B, S1, Sum)
    ).

:- pred once(int::in, array(int)::array_di, array(int)::array_uo) is det.
once(Lo, !A) :-
    !A ^ unsafe_elem(Lo) := 99.

:- pred show(array(int)::array_ui, io::di, io::uo) is det.
show(A, !IO) :-
    list.map(int_to_string, array.to_list(A), Strings),
    io.write_string(string.join_list(" ", Strings), !IO),
    io.nl(!IO).

main(!IO) :-
    array.init(20, 0, A0),
    fill(1, 19, 1, 10, A0, A1),
    show(A1, !IO),
    fill(0, 19, -1, 100, A1, A2),
    show(A2, !IO),
    once(5, A2, A3),
    show(A3, !IO),
    N = 4,
    M = array.generate(N * N, (func(X) = X + 1)),
    D = array.generate(N * N, (func(X) = (if X mod (N + 1) = 0 then 2 else 0))),
    array.init(N * N, 0, C0),
    matmul(N, M, D, C0, C),
    show(C, !IO).
