% Abstract types that are equivalences, for hidden-equivalences.m: of a
% float, which passes in a float register, of a narrow integer, of a
% polymorphic list whose module only the implementation section imports,
% and of another of them.

:- module hidden_equivalences.
:- interface.
:- import_module string.

:- type weight.
:- type small.
:- type bag(T).
:- type heavy.

:- func weight(float) = weight.
:- func add(weight, weight) = weight.
:- func to_float(weight) = float.
:- func small(int) = small.
:- func next(small) = small.
:- func bag(T) = bag(T).
:- func put(T, bag(T)) = bag(T).
:- func size(bag(T)) = int.
:- func heavy(weight) = heavy.
:- func describe(heavy) = string.

:- implementation.
:- import_module float, int, list, uint8.

:- type weight == float.
:- type small == uint8.
:- type bag(T) == list({T, int}).
:- type heavy == weight.

weight(F) = F.
add(A, B) = A + B.
to_float(W) = W.
small(N) = uint8.cast_from_int(N).
next(S) = S + 1u8.
bag(X) = [{X, 1}].
put(X, B) = [{X, list.length(B) + 1} | B].
size(B) = list.length(B).
heavy(W) = W * 10.0.
describe(H) = string.float_to_string(H).
