% The translation of run-ground-tuples.mt for mmc: the open field ground.
:- module ground_tuples_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string.

:- type job
    --->    job(list(int), {int, string}, {list(int), {@T@, int}}).

:- pred describe(job::in, string::out) is det.

describe(job(_, Named, {Ws, Pair}), Text) :-
    named(Named, N, Name),
    pair(Pair, F, K),
    Text = string.format("%s: %d, %d weights, %.1f, %d",
        [s(Name), i(N), i(list.length(Ws)), f(@TO_FLOAT@(F)), i(K)]).

    % Whole tuples, passed as in.
:- pred named({int, string}::in, int::out, string::out) is det.

named({N, Name}, N, Name).

:- pred pair({@T@, int}::in, @T@::out, int::out) is det.

pair({F, K}, F, K).

:- pred make(list(int)::in, job::out) is det.

make(Holes, job(Holes, {3, "three"}, {[1, 2], {0.5@S@, 7}})).

main(!IO) :-
    make([0, 0], Job),
    describe(Job, Text),
    io.write_string(Text ++ "\n", !IO).
