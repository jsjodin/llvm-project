% A module of the program's own, besides the library, whose procedure a
% target region calls (omp-target-other-module.m).
:- module omp_target_other_module.
:- interface.
:- func triple(int) = int.
:- implementation.
:- import_module int.

triple(X) = 3 * X.
