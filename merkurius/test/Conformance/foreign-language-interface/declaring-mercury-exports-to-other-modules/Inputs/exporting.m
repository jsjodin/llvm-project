% A module that exports a function to C.
:- module exporting.
:- interface.
:- func triple(int) = int.
:- implementation.
:- import_module int.
triple(X) = 3 * X.
:- pragma foreign_export("C", triple(in) = out, "exporting_triple").
