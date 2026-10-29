% The running example of the design document (Front end, Running example).

:- module sumsq.
:- interface.
:- import_module array, int.

:- pred sum_squares(array(int)::in, int::in, int::out) is det.

:- implementation.

sum_squares(A, N, Sum) :-
    %$omp parallel do reduction(+)
    int.fold_up(add_square(A), 0, N, 0, Sum).

:- pred add_square(array(int)::in, int::in, int::in, int::out) is det.

add_square(A, I, !Acc) :-
    array.unsafe_lookup(A, I, X),
    !:Acc = !.Acc + X * X.
