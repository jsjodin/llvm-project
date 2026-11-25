% The feature and inner context of run-flat-values.m.
:- module nested_host_process_do_loop_flat_values_lib.
:- interface.
:- import_module int8, string, uint16.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag,
    % which Merkurius holds by value.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, float).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module float, int, list.

make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    float(N) * 0.25).

use(fdata(C, wrapped(W), I, U, F)) =
    string.string(C) ++ " " ++ string.int_to_string(W) ++ " " ++
    string.int_to_string(int8.to_int(I)) ++ " " ++
    string.int_to_string(uint16.to_int(U)) ++ " " ++
    string.float_to_string(F).

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.

inner_run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make(K),
        Ss1 = [use(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make(N), V0, V1, Last) do
        ( if I > 0 then V1 = make(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use(Last).
