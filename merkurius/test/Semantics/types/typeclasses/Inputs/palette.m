% A module whose interface imports colours, for the modules that import
% this one and not colours.

:- module palette.
:- interface.
:- import_module colours.
:- func favourite = colour.
:- implementation.
favourite = green.
