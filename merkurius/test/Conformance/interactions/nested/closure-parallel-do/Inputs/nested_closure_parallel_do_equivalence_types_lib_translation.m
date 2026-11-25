% The feature and inner context of run-equivalence-types.m.
:- module nested_closure_parallel_do_equivalence_types_lib_translation.
:- interface.
:- import_module list, pair, string.

:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module array, int.

make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).


inner_run(N) = string.join_list(" | ",
    list.map((func(I) = use(make(N + I))), 0 .. 3)).
