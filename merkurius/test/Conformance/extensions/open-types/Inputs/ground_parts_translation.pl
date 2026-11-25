% The translation of run-ground-parts.m into Prolog.
:- ensure_loaded(mercury_write).

describe(job(_, N, Name, Weights), Text) :-
    length(Weights, Count),
    sum_list(Weights, Sum),
    format(string(Text), "~s: ~d, ~d weights summing to ~d",
        [Name, N, Count, Sum]).

make(Holes, job(Holes, 3, "three", [1, 2, 3])).

main :-
    L = [_, _],
    make(L, Job),
    describe(Job, Text),
    format("~s~n", [Text]).
