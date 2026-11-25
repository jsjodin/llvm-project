:- module include_parent.helpers.
:- interface.
:- import_module list, include_parent.shapes.
:- func total(list(shape)) = int.
:- implementation.
:- import_module int.
total(Shapes) = list.foldl((func(S, A) = A + area(S)), Shapes, 0).
