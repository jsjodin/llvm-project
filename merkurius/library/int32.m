% A declaration-only stub of the standard library module int32, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module int32.
:- interface.
:- import_module int16.
:- import_module int64.
:- import_module int8.
:- import_module uint32.
:- import_module uint8.

:- pred from_int(int::in, int32::out) is semidet.
:- func det_from_int(int) = int32.
:- func cast_from_int(int) = int32.
:- func to_int(int32) = int.
:- func cast_to_int(int32) = int.
:- func cast_to_int8(int32) = int8.
:- func cast_from_int8(int8) = int32.
:- func cast_to_int16(int32) = int16.
:- func cast_from_int16(int16) = int32.
:- func cast_to_int64(int32) = int64.
:- func cast_from_int64(int64) = int32.
:- func cast_from_uint32(uint32) = int32.
:- func from_bytes_le(uint8, uint8, uint8, uint8) = int32.
:- func from_bytes_be(uint8, uint8, uint8, uint8) = int32.
:- pred (int32::in) < (int32::in) is semidet.
:- pred (int32::in) > (int32::in) is semidet.
:- pred (int32::in) =< (int32::in) is semidet.
:- pred (int32::in) >= (int32::in) is semidet.
:- func max(int32, int32) = int32.
:- func min(int32, int32) = int32.
:- func abs(int32) = int32.
:- func unchecked_abs(int32) = int32.
:- func nabs(int32) = int32.
:- func + (int32::in) = (int32::uo) is det.
:- func - (int32::in) = (int32::uo) is det.
:- func int32 + int32 = int32.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(int32, int32) = int32.
:- func int32 - int32 = int32.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(int32, int32) = int32.
:- func (int32::in) * (int32::in) = (int32::uo) is det.
:- func times(int32, int32) = int32.
:- func (int32::in) div (int32::in) = (int32::uo) is det.
:- func (int32::in) // (int32::in) = (int32::uo) is det.
:- func (int32::in) / (int32::in) = (int32::uo) is det.
:- func unchecked_quotient(int32::in, int32::in) = (int32::uo) is det.
:- func (int32::in) mod (int32::in) = (int32::uo) is det.
:- func (int32::in) rem (int32::in) = (int32::uo) is det.
:- func unchecked_rem(int32::in, int32::in) = (int32::uo) is det.
:- pred even(int32::in) is semidet.
:- pred odd(int32::in) is semidet.
:- func (int32::in) << (int::in) = (int32::uo) is det.
:- func unchecked_left_shift(int32::in, int::in) = (int32::uo) is det.
:- func (int32::in) >> (int::in) = (int32::uo) is det.
:- func unchecked_right_shift(int32::in, int::in) = (int32::uo) is det.
:- func (int32::in) /\ (int32::in) = (int32::uo) is det.
:- func (int32::in) \/ (int32::in) = (int32::uo) is det.
:- func xor(int32, int32) = int32.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (int32::in) = (int32::uo) is det.
:- func num_zeros(int32) = int.
:- func num_ones(int32) = int.
:- func num_leading_zeros(int32) = int.
:- func num_trailing_zeros(int32) = int.
:- func reverse_bytes(int32) = int32.
:- func reverse_bits(int32) = int32.
:- func min_int32 = int32.
:- func max_int32 = int32.
