% A declaration-only stub of the standard library module int16, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module int16.
:- interface.
:- import_module uint16.
:- import_module uint8.

:- pred from_int(int::in, int16::out) is semidet.
:- func det_from_int(int) = int16.
:- func cast_from_int(int) = int16.
:- func to_int(int16) = int.
:- func cast_to_int(int16) = int.
:- func cast_from_uint16(uint16) = int16.
:- func from_bytes_le(uint8, uint8) = int16.
:- func from_bytes_be(uint8, uint8) = int16.
:- pred (int16::in) < (int16::in) is semidet.
:- pred (int16::in) > (int16::in) is semidet.
:- pred (int16::in) =< (int16::in) is semidet.
:- pred (int16::in) >= (int16::in) is semidet.
:- func max(int16, int16) = int16.
:- func min(int16, int16) = int16.
:- func abs(int16) = int16.
:- func unchecked_abs(int16) = int16.
:- func nabs(int16) = int16.
:- func + (int16::in) = (int16::uo) is det.
:- func - (int16::in) = (int16::uo) is det.
:- func int16 + int16 = int16.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(int16, int16) = int16.
:- func int16 - int16 = int16.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(int16, int16) = int16.
:- func (int16::in) * (int16::in) = (int16::uo) is det.
:- func times(int16, int16) = int16.
:- func (int16::in) div (int16::in) = (int16::uo) is det.
:- func (int16::in) // (int16::in) = (int16::uo) is det.
:- func (int16::in) / (int16::in) = (int16::uo) is det.
:- func unchecked_quotient(int16::in, int16::in) = (int16::uo) is det.
:- func (int16::in) mod (int16::in) = (int16::uo) is det.
:- func (int16::in) rem (int16::in) = (int16::uo) is det.
:- func unchecked_rem(int16::in, int16::in) = (int16::uo) is det.
:- pred even(int16::in) is semidet.
:- pred odd(int16::in) is semidet.
:- func (int16::in) << (int::in) = (int16::uo) is det.
:- func unchecked_left_shift(int16::in, int::in) = (int16::uo) is det.
:- func (int16::in) >> (int::in) = (int16::uo) is det.
:- func unchecked_right_shift(int16::in, int::in) = (int16::uo) is det.
:- func (int16::in) /\ (int16::in) = (int16::uo) is det.
:- func (int16::in) \/ (int16::in) = (int16::uo) is det.
:- func xor(int16, int16) = int16.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (int16::in) = (int16::uo) is det.
:- func num_zeros(int16) = int.
:- func num_ones(int16) = int.
:- func num_leading_zeros(int16) = int.
:- func num_trailing_zeros(int16) = int.
:- func reverse_bytes(int16) = int16.
:- func reverse_bits(int16) = int16.
:- func min_int16 = int16.
:- func max_int16 = int16.
