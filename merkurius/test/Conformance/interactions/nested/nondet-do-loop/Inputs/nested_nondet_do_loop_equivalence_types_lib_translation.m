% The feature and inner context of run-equivalence-types.m.
:- module nested_nondet_do_loop_equivalence_types_lib_translation.
:- interface.
:- import_module list, pair, string.

:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int.

make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).


inner_run(N) = string.join_list(" | ",
    list.map((func(K) = use(make(K))), [N, N + 1])) ++ " | last " ++
    use(make(N + 20)).
