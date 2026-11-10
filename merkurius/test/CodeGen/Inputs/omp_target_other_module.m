% A module of the program's own, besides the library: its procedures are
% not compiled for a device yet (omp-target-other-module.m).
:- module omp_target_other_module.
:- interface.
:- func triple(int) = int.
:- implementation.
:- import_module int.

triple(X) = 3 * X.
