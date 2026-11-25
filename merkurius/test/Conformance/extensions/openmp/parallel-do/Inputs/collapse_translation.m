% The translation of nyi-collapse.m's loops, without their directive.
:- module collapse_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, string.

:- pred matmul(int::in, array(float)::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
matmul(N, A, B, !C) :-
    rows(0, N, A, B, !C).

:- pred rows(int::in, int::in, array(float)::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
rows(I, N, A, B, !C) :-
    ( if I > N - 1 then
        true
    else
        columns(I, 0, N, A, B, !C),
        rows(I + 1, N, A, B, !C)
    ).

:- pred columns(int::in, int::in, int::in, array(float)::in,
    array(float)::in, array(float)::array_di, array(float)::array_uo) is det.
columns(I, J, N, A, B, !C) :-
    ( if J > N - 1 then
        true
    else
        dot(I, J, 0, N, A, B, 0.0, Sum),
        !C ^ unsafe_elem(I * N + J) := Sum,
        columns(I, J + 1, N, A, B, !C)
    ).

:- pred dot(int::in, int::in, int::in, int::in, array(float)::in,
    array(float)::in, float::in, float::out) is det.
dot(I, J, K, N, A, B, S0, Sum) :-
    ( if K > N - 1 then
        Sum = S0
    else
        S1 = S0 + A ^ unsafe_elem(I * N + K) * B ^ unsafe_elem(K * N + J),
        dot(I, J, K + 1, N, A, B, S1, Sum)
    ).

main(!IO) :-
    N = 3,
    A = array.generate(N * N, (func(X) = float(X))),
    B = array.generate(N * N, (func(X) = (if X mod (N + 1) = 0 then 1.0 else 0.0))),
    array.init(N * N, 0.0, C0),
    matmul(N, A, B, C0, C),
    io.write_string(string.join_list(" ",
        list.map(float_to_string, array.to_list(C))), !IO),
    io.nl(!IO).
