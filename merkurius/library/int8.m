% A declaration-only stub of the standard library module int8, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module int8.
:- interface.
:- import_module uint8.

:- pred from_int(int::in, int8::out) is semidet.
:- func det_from_int(int) = int8.
:- func cast_from_int(int) = int8.
:- func to_int(int8) = int.
:- func cast_to_int(int8) = int.
:- func cast_from_uint8(uint8) = int8.
:- pred (int8::in) < (int8::in) is semidet.
:- pred (int8::in) > (int8::in) is semidet.
:- pred (int8::in) =< (int8::in) is semidet.
:- pred (int8::in) >= (int8::in) is semidet.
:- func max(int8, int8) = int8.
:- func min(int8, int8) = int8.
:- func abs(int8) = int8.
:- func unchecked_abs(int8) = int8.
:- func nabs(int8) = int8.
:- func + (int8::in) = (int8::uo) is det.
:- func - (int8::in) = (int8::uo) is det.
:- func int8 + int8 = int8.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(int8, int8) = int8.
:- func int8 - int8 = int8.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(int8, int8) = int8.
:- func (int8::in) * (int8::in) = (int8::uo) is det.
:- func times(int8, int8) = int8.
:- func (int8::in) div (int8::in) = (int8::uo) is det.
:- func (int8::in) // (int8::in) = (int8::uo) is det.
:- func (int8::in) / (int8::in) = (int8::uo) is det.
:- func unchecked_quotient(int8::in, int8::in) = (int8::uo) is det.
:- func (int8::in) mod (int8::in) = (int8::uo) is det.
:- func (int8::in) rem (int8::in) = (int8::uo) is det.
:- func unchecked_rem(int8::in, int8::in) = (int8::uo) is det.
:- pred even(int8::in) is semidet.
:- pred odd(int8::in) is semidet.
:- func (int8::in) << (int::in) = (int8::uo) is det.
:- func unchecked_left_shift(int8::in, int::in) = (int8::uo) is det.
:- func (int8::in) >> (int::in) = (int8::uo) is det.
:- func unchecked_right_shift(int8::in, int::in) = (int8::uo) is det.
:- func (int8::in) /\ (int8::in) = (int8::uo) is det.
:- func (int8::in) \/ (int8::in) = (int8::uo) is det.
:- func xor(int8, int8) = int8.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (int8::in) = (int8::uo) is det.
:- func num_zeros(int8) = int.
:- func num_ones(int8) = int.
:- func num_leading_zeros(int8) = int.
:- func num_trailing_zeros(int8) = int.
:- func reverse_bits(int8) = int8.
:- func min_int8 = int8.
:- func max_int8 = int8.
