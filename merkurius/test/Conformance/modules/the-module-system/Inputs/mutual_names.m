:- module mutual_names.
:- interface.
:- import_module mutual_colours.
:- func describe(colour) = string.
:- implementation.
describe(red) = "red".
describe(green) = "green".
