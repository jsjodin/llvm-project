% A declaration-only stub of the standard library module uint64, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module uint64.
:- interface.
:- import_module int64.
:- import_module uint8.

:- pred from_int(int::in, uint64::out) is semidet.
:- func det_from_int(int) = uint64.
:- func cast_from_int(int) = uint64.
:- func cast_from_uint(uint) = uint64.
:- func cast_to_int(uint64) = int.
:- func cast_to_uint(uint64) = uint.
:- func cast_to_uint8(uint64) = uint8.
:- func cast_from_uint8(uint8) = uint64.
:- func cast_from_int64(int64) = uint64.
:- func from_bytes_le(uint8, uint8, uint8, uint8, uint8, uint8, uint8, uint8)
    = uint64.
:- func from_bytes_be(uint8, uint8, uint8, uint8, uint8, uint8, uint8, uint8)
    = uint64.
:- pred (uint64::in) < (uint64::in) is semidet.
:- pred (uint64::in) > (uint64::in) is semidet.
:- pred (uint64::in) =< (uint64::in) is semidet.
:- pred (uint64::in) >= (uint64::in) is semidet.
:- func max(uint64, uint64) = uint64.
:- func min(uint64, uint64) = uint64.
:- func uint64 + uint64 = uint64.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(uint64, uint64) = uint64.
:- func uint64 - uint64 = uint64.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(uint64, uint64) = uint64.
:- func (uint64::in) * (uint64::in) = (uint64::uo) is det.
:- func times(uint64, uint64) = uint64.
:- func (uint64::in) div (uint64::in) = (uint64::uo) is det.
:- func (uint64::in) // (uint64::in) = (uint64::uo) is det.
:- func (uint64::in) / (uint64::in) = (uint64::uo) is det.
:- func unchecked_quotient(uint64::in, uint64::in) = (uint64::uo) is det.
:- func (uint64::in) mod (uint64::in) = (uint64::uo) is det.
:- func (uint64::in) rem (uint64::in) = (uint64::uo) is det.
:- func unchecked_rem(uint64::in, uint64::in) = (uint64::uo) is det.
:- pred even(uint64::in) is semidet.
:- pred odd(uint64::in) is semidet.
:- func (uint64::in) << (int::in) = (uint64::uo) is det.
:- func unchecked_left_shift(uint64::in, int::in) = (uint64::uo) is det.
:- func (uint64::in) >> (int::in) = (uint64::uo) is det.
:- func unchecked_right_shift(uint64::in, int::in) = (uint64::uo) is det.
:- func (uint64::in) /\ (uint64::in) = (uint64::uo) is det.
:- func (uint64::in) \/ (uint64::in) = (uint64::uo) is det.
:- func xor(uint64, uint64) = uint64.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (uint64::in) = (uint64::uo) is det.
:- func num_zeros(uint64) = int.
:- func num_ones(uint64) = int.
:- func num_leading_zeros(uint64) = int.
:- func num_trailing_zeros(uint64) = int.
:- func reverse_bytes(uint64) = uint64.
:- func reverse_bits(uint64) = uint64.
:- func rotate_left(uint64, uint) = uint64.
:- func unchecked_rotate_left(uint64, uint) = uint64.
:- func rotate_right(uint64, uint) = uint64.
:- func unchecked_rotate_right(uint64, uint) = uint64.
:- func set_bit(uint64, uint) = uint64.
:- func unchecked_set_bit(uint64, uint) = uint64.
:- func clear_bit(uint64, uint) = uint64.
:- func unchecked_clear_bit(uint64, uint) = uint64.
:- func flip_bit(uint64, uint) = uint64.
:- func unchecked_flip_bit(uint64, uint) = uint64.
:- pred bit_is_set(uint64::in, uint::in) is semidet.
:- pred unchecked_bit_is_set(uint64::in, uint::in) is semidet.
:- pred bit_is_clear(uint64::in, uint::in) is semidet.
:- pred unchecked_bit_is_clear(uint64::in, uint::in) is semidet.
:- func max_uint64 = uint64.
