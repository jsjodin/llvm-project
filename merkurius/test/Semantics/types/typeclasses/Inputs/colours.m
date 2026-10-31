% An instance of an imported type class, for this module's type.

:- module colours.
:- interface.
:- import_module classes.
:- type colour ---> red ; green.
:- instance shown(colour).
:- implementation.
:- instance shown(colour) where [ shown(_) = "colour" ].
