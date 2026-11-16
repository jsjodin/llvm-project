% A module of the program's own, besides the library, whose procedures GPU
% code calls (process-other-module.m): a type of its own, a procedure its
% interface leaves out, recursion, and calls of the library.
:- module device_helpers.
:- interface.
:- import_module uint64.

:- type shape
    --->    square(int)
    ;       rect(int, int).

:- func area(shape) = int.

    % The steps of the 3n + 1 sequence from N down to 1.
:- func steps(int) = int.

    % The top 24 bits of X, through a float, as the library converts it.
:- func top_bits(uint64) = int.

:- implementation.
:- import_module float, int.

area(square(S)) = S * S.
area(rect(W, H)) = W * H.

steps(N) = ( if N =< 1 then 0 else 1 + steps(next(N)) ).

:- func next(int) = int.

next(N) = ( if N mod 2 = 0 then N / 2 else 3 * N + 1 ).

top_bits(X) = float.truncate_to_int(float.cast_from_uint64(X >> 40)).
