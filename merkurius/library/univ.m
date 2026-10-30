% A declaration-only stub of the standard library module univ, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module univ.
:- interface.

:- type univ.
:- pred type_to_univ(T, univ).
:- mode type_to_univ(di, uo) is det.
:- mode type_to_univ(in, out) is det.
:- mode type_to_univ(out, in) is semidet.
:- pred univ_to_type(univ, T).
:- mode univ_to_type(in, out) is semidet.
:- mode univ_to_type(out, in) is det.
:- mode univ_to_type(uo, di) is det.
:- func univ(T) = univ.
:- mode univ(in) = out is det.
:- mode univ(di) = uo is det.
:- mode univ(out) = in is semidet.
:- pred det_univ_to_type(univ::in, T::out) is det.
:- some [T] func univ_value(univ) = T.
