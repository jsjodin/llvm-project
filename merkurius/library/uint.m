% A declaration-only stub of the standard library module uint, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module uint.
:- interface.

:- pred from_int(int::in, uint::out) is semidet.
:- func det_from_int(int) = uint.
:- func cast_from_int(int) = uint.
:- func cast_to_int(uint) = int.
:- pred (uint::in) < (uint::in) is semidet.
:- pred (uint::in) > (uint::in) is semidet.
:- pred (uint::in) =< (uint::in) is semidet.
:- pred (uint::in) >= (uint::in) is semidet.
:- func max(uint, uint) = uint.
:- func min(uint, uint) = uint.
:- func uint + uint = uint.
:- mode in   + in  = uo is det.
:- mode uo   + in  = in is det.
:- mode in   + uo  = in is det.
:- func plus(uint, uint) = uint.
:- func uint - uint = uint.
:- mode in   - in   = uo is det.
:- mode uo   - in   = in is det.
:- mode in   - uo   = in is det.
:- func minus(uint, uint) = uint.
:- func (uint::in) * (uint::in) = (uint::uo) is det.
:- func times(uint, uint) = uint.
:- func (uint::in) div (uint::in) = (uint::uo) is det.
:- func (uint::in) // (uint::in) = (uint::uo) is det.
:- func (uint::in) / (uint::in) = (uint::uo) is det.
:- func unchecked_quotient(uint::in, uint::in) = (uint::uo) is det.
:- func (uint::in) mod (uint::in) = (uint::uo) is det.
:- func (uint::in) rem (uint::in) = (uint::uo) is det.
:- func unchecked_rem(uint::in, uint::in) = (uint::uo) is det.
:- func (uint::in) << (int::in) = (uint::uo) is det.
:- func unchecked_left_shift(uint::in, int::in) = (uint::uo) is det.
:- func (uint::in) >> (int::in) = (uint::uo) is det.
:- func unchecked_right_shift(uint::in, int::in) = (uint::uo) is det.
:- pred even(uint::in) is semidet.
:- pred odd(uint::in) is semidet.
:- func (uint::in) /\ (uint::in) = (uint::uo) is det.
:- func (uint::in) \/ (uint::in) = (uint::uo) is det.
:- func xor(uint, uint) = uint.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (uint::in) = (uint::uo) is det.
:- func max_uint = uint.
:- func bits_per_uint = int.
:- func hash(uint) = int.
:- pred hash(uint::in, int::out) is det.
