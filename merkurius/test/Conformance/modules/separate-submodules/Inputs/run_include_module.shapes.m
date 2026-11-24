:- module run_include_module.shapes.
:- interface.
:- type shape
    --->    square(int)
    ;       circle(int).
:- func area(shape) = int.
:- implementation.
:- import_module int.
area(square(S)) = S * S.
area(circle(R)) = scale * R * R.
:- end_module shapes.
