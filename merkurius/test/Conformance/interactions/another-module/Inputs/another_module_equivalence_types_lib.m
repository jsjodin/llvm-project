% The feature of run-equivalence-types.m, as another module.
:- module another_module_equivalence_types_lib.
:- interface.
:- import_module list, pair.

:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- implementation.
:- import_module int, string.

make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).
