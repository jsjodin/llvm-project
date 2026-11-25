% The translation of Inputs/another_module_open_types_lib.m for mmc.
:- module another_module_open_types_lib_translation.
:- interface.
:- import_module list.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- implementation.
:- import_module int, string.

make(N) = (1 .. N) ++ [N * 10].

use(L) =
    ( if L = [A, B | Rest] then
        string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        "short"
    ).
