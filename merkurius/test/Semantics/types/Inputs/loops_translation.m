% The translation of each loop of ../loops.m into a recursive predicate,
% written by hand as the design document's "Loops" gives it, with the
% types that merkc gives the loops' variables declared. mmc accepts it, as
% det.

:- module loops_translation.
:- interface.
:- import_module array, list, pair.

:- pred sum_list(list(int)::in, int::in, int::out) is det.
:- pred pi(int::in, float::out) is det.
:- pred sum_floats(list(float)::in, float::out) is det.
:- pred keys(list(pair(string, int))::in, list(string)::out, int::out) is det.
:- pred saxpy(float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
:- pred triangle(int::in, int::out) is det.
:- pred inferred(list(int)::in, int::out, int::out) is det.

:- implementation.
:- import_module float, int.

sum_list(Xs, !Sum) :-
    sum_list_loop(Xs, !Sum).

:- pred sum_list_loop(list(int)::in, int::in, int::out) is det.
sum_list_loop(L, !Sum) :-
    (
        L = []
    ;
        L = [X | T],
        !:Sum = !.Sum + X,
        sum_list_loop(T, !Sum)
    ).

pi(N, Pi) :-
    Step = 1.0 / float(N),
    pi_loop(0, N - 1, Step, 0.0, Sum),
    Pi = Step * Sum.

:- pred pi_loop(int::in, int::in, float::in, float::in, float::out) is det.
pi_loop(I, Hi, Step, S0, Sum) :-
    ( if I > Hi then
        Sum = S0
    else
        X = (float(I) + 0.5) * Step,
        Y = 4.0 / (1.0 + X * X),
        S1 = S0 + Y,
        pi_loop(I + 1, Hi, Step, S1, Sum)
    ).

sum_floats(Xs, Sum) :-
    sum_floats_loop(Xs, 0.0, Sum).

:- pred sum_floats_loop(list(float)::in, float::in, float::out) is det.
sum_floats_loop(L, S0, Sum) :-
    (
        L = [],
        Sum = S0
    ;
        L = [X | T],
        S1 = S0 + X,
        sum_floats_loop(T, S1, Sum)
    ).

keys(Pairs, Keys, Total) :-
    keys_loop(Pairs, Keys, 0, Total).

:- pred keys_loop(list(pair(string, int))::in, list(string)::out,
    int::in, int::out) is det.
keys_loop(Pairs, Keys, T0, Total) :-
    (
        Pairs = [],
        Keys = [],
        Total = T0
    ;
        Pairs = [K - V | Ps],
        Keys = [K | Ks],
        T1 = T0 + V,
        keys_loop(Ps, Ks, T1, Total)
    ).

saxpy(A, X, !Y) :-
    N = array.size(X),
    saxpy_loop(0, N - 1, A, X, !Y).

:- pred saxpy_loop(int::in, int::in, float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
saxpy_loop(I, Hi, A, X, !Y) :-
    ( if I > Hi then
        true
    else
        !Y ^ unsafe_elem(I) := A * X ^ unsafe_elem(I) + !.Y ^ unsafe_elem(I),
        saxpy_loop(I + 1, Hi, A, X, !Y)
    ).

triangle(N, Total) :-
    triangle_loop_1(1, N, 0, Total).

:- pred triangle_loop_1(int::in, int::in, int::in, int::out) is det.
triangle_loop_1(I, Hi, T0, Total) :-
    ( if I > Hi then
        Total = T0
    else
        triangle_loop_2(1, I, I, T0, T1),
        triangle_loop_1(I + 1, Hi, T1, Total)
    ).

:- pred triangle_loop_2(int::in, int::in, int::in, int::in, int::out)
    is det.
triangle_loop_2(J, Hi, I, U0, T1) :-
    ( if J > Hi then
        T1 = U0
    else
        U1 = U0 + I * J,
        triangle_loop_2(J + 1, Hi, I, U1, T1)
    ).

inferred(Xs, Positive, Length) :-
    count_positive(Xs, Positive),
    length_of(Xs, Length).

count_positive(Xs, N) :-
    count_positive_loop(Xs, 0, N).

:- pred count_positive_loop(list(int)::in, int::in, int::out) is det.
count_positive_loop(L, C0, N) :-
    (
        L = [],
        N = C0
    ;
        L = [X | T],
        ( if X > 0 then C1 = C0 + 1 else C1 = C0 ),
        count_positive_loop(T, C1, N)
    ).

% count(_, 1, N): the counter's next value, with N the last one.
length_of(Xs, N) :-
    length_of_loop(Xs, 1, N).

:- pred length_of_loop(list(T)::in, int::in, int::out) is det.
length_of_loop(L, K, N) :-
    (
        L = [],
        N = K - 1
    ;
        L = [_ | T],
        length_of_loop(T, K + 1, N)
    ).
