% The feature and inner context of run-open-types.m.
:- module nested_do_loop_parallel_do_open_types_lib_translation.
:- interface.
:- import_module list, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module array, int.

make(N) = (1 .. N) ++ [N * 10].

use(L) =
    ( if L = [A, B | Rest] then
        string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        "short"
    ).


inner_run(N) = string.join_list(" | ",
    list.map((func(I) = use(make(N + I))), 0 .. 3)).
