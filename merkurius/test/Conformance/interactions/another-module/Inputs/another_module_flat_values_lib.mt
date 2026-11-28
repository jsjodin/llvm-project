% The feature of run-flat-values.mt, as another module.
:- module another_module_flat_values_lib.
:- interface.
:- import_module @TM@, int8, uint16.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag,
    % which Merkurius holds by value.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, @T@).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- implementation.
:- import_module int, string.

make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    @FROM_INT@(N) * 0.25@S@).

use(fdata(C, wrapped(W), I, U, F)) =
    string.string(C) ++ " " ++ string.int_to_string(W) ++ " " ++
    string.int_to_string(int8.to_int(I)) ++ " " ++
    string.int_to_string(uint16.to_int(U)) ++ " " ++
    string.@T@_to_string(F).
