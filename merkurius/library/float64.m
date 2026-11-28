% The library module float64, a Merkurius extension (docs/INPROGRESS.md,
% Sized floating point types): float64 is float, IEEE double precision,
% named by its size as float32 and the sized integers are. It is an
% equivalence type, so every function of the module float applies to a
% float64, and a float64 is a float wherever one is wanted. Not in the
% library reference manual or in mmc.

:- module float64.
:- interface.
:- import_module float.

:- type float64 == float.
