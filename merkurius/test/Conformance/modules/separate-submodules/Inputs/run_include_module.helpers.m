:- module run_include_module.helpers.
:- interface.
:- import_module list, run_include_module.shapes.
:- func total(list(shape)) = int.
:- implementation.
:- import_module int.
total(Shapes) = list.foldl((func(S, A) = A + area(S)), Shapes, 0).
