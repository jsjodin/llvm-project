:- module mutual_colours.
:- interface.
:- import_module mutual_names.
:- type colour
    --->    red
    ;       green.
:- func name(colour) = string.
:- implementation.
name(C) = describe(C).
