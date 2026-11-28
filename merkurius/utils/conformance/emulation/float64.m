% float64 for mmc, which has none: the oracle's stand-in for Merkurius'
% library/float64.m, which oracle.py copies beside a test that imports it.
% float64 is float there as here, so the stand-in is the library module.

:- module float64.
:- interface.
:- import_module float.

:- type float64 == float.
