% Loops laid out carelessly (design document, "Loops").
:- module loops_messy.
:- implementation.
below_sum(N, Count) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do S1 = S0 + I ),
  ( for(I, 1, N),
      fromto(0, C0, C1, Count) do ( if I * I < Sum then C1 = C0 + 1 else C1 = C0 ) ).
sum_list(Xs, !S) :- ( foreach(X, Xs) do !:S = !.S + X ).
nest(N, !C) :-
    %$omp parallel do collapse(2)
    ( for(I, 0, N - 1) do ( for(J, 0, N - 1) do
        !C ^ unsafe_elem(I * N + J) := I + J ) ).
