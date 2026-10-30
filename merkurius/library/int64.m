% A declaration-only stub of the standard library module int64, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module int64.
:- interface.
:- import_module uint64.
:- import_module uint8.

:- func from_int(int) = int64.
:- func cast_from_int(int) = int64.
:- pred to_int(int64::in, int::out) is semidet.
:- func det_to_int(int64) = int.
:- func cast_to_int(int64) = int.
:- func cast_from_uint64(uint64) = int64.
:- func from_bytes_le(uint8, uint8, uint8, uint8, uint8, uint8, uint8, uint8)
    = int64.
:- func from_bytes_be(uint8, uint8, uint8, uint8, uint8, uint8, uint8, uint8)
    = int64.
:- pred (int64::in) < (int64::in) is semidet.
:- pred (int64::in) > (int64::in) is semidet.
:- pred (int64::in) =< (int64::in) is semidet.
:- pred (int64::in) >= (int64::in) is semidet.
:- func max(int64, int64) = int64.
:- func min(int64, int64) = int64.
:- func abs(int64) = int64.
:- func unchecked_abs(int64) = int64.
:- func nabs(int64) = int64.
:- func + (int64::in) = (int64::uo) is det.
:- func - (int64::in) = (int64::uo) is det.
:- func int64 + int64 = int64.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(int64, int64) = int64.
:- func int64 - int64 = int64.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(int64, int64) = int64.
:- func (int64::in) * (int64::in) = (int64::uo) is det.
:- func times(int64, int64) = int64.
:- func (int64::in) div (int64::in) = (int64::uo) is det.
:- func (int64::in) // (int64::in) = (int64::uo) is det.
:- func (int64::in) / (int64::in) = (int64::uo) is det.
:- func unchecked_quotient(int64::in, int64::in) = (int64::uo) is det.
:- func (int64::in) mod (int64::in) = (int64::uo) is det.
:- func (int64::in) rem (int64::in) = (int64::uo) is det.
:- func unchecked_rem(int64::in, int64::in) = (int64::uo) is det.
:- pred even(int64::in) is semidet.
:- pred odd(int64::in) is semidet.
:- func (int64::in) << (int::in) = (int64::uo) is det.
:- func unchecked_left_shift(int64::in, int::in) = (int64::uo) is det.
:- func (int64::in) >> (int::in) = (int64::uo) is det.
:- func unchecked_right_shift(int64::in, int::in) = (int64::uo) is det.
:- func (int64::in) /\ (int64::in) = (int64::uo) is det.
:- func (int64::in) \/ (int64::in) = (int64::uo) is det.
:- func xor(int64, int64) = int64.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (int64::in) = (int64::uo) is det.
:- func num_zeros(int64) = int.
:- func num_ones(int64) = int.
:- func num_leading_zeros(int64) = int.
:- func num_trailing_zeros(int64) = int.
:- func reverse_bytes(int64) = int64.
:- func reverse_bits(int64) = int64.
:- func min_int64 = int64.
:- func max_int64 = int64.
