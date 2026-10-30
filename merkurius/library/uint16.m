% A declaration-only stub of the standard library module uint16, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module uint16.
:- interface.
:- import_module int16.
:- import_module uint64.
:- import_module uint8.

:- pred from_int(int::in, uint16::out) is semidet.
:- func det_from_int(int) = uint16.
:- func cast_from_int(int) = uint16.
:- pred from_uint(uint::in, uint16::out) is semidet.
:- func det_from_uint(uint) = uint16.
:- func cast_from_uint(uint) = uint16.
:- func to_int(uint16) = int.
:- func cast_to_int(uint16) = int.
:- func cast_to_uint(uint16) = uint.
:- func cast_to_uint8(uint16) = uint8.
:- func cast_from_uint8(uint8) = uint16.
:- func cast_to_uint64(uint16) = uint64.
:- func cast_from_uint64(uint64) = uint16.
:- func cast_from_int16(int16) = uint16.
:- func from_bytes_le(uint8, uint8) = uint16.
:- func from_bytes_be(uint8, uint8) = uint16.
:- pred (uint16::in) < (uint16::in) is semidet.
:- pred (uint16::in) > (uint16::in) is semidet.
:- pred (uint16::in) =< (uint16::in) is semidet.
:- pred (uint16::in) >= (uint16::in) is semidet.
:- func max(uint16, uint16) = uint16.
:- func min(uint16, uint16) = uint16.
:- func uint16 + uint16 = uint16.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(uint16, uint16) = uint16.
:- func uint16 - uint16 = uint16.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(uint16, uint16) = uint16.
:- func (uint16::in) * (uint16::in) = (uint16::uo) is det.
:- func times(uint16, uint16) = uint16.
:- func (uint16::in) div (uint16::in) = (uint16::uo) is det.
:- func (uint16::in) // (uint16::in) = (uint16::uo) is det.
:- func (uint16::in) / (uint16::in) = (uint16::uo) is det.
:- func unchecked_quotient(uint16::in, uint16::in) = (uint16::uo) is det.
:- func (uint16::in) mod (uint16::in) = (uint16::uo) is det.
:- func (uint16::in) rem (uint16::in) = (uint16::uo) is det.
:- func unchecked_rem(uint16::in, uint16::in) = (uint16::uo) is det.
:- pred even(uint16::in) is semidet.
:- pred odd(uint16::in) is semidet.
:- func (uint16::in) << (int::in) = (uint16::uo) is det.
:- func unchecked_left_shift(uint16::in, int::in) = (uint16::uo) is det.
:- func (uint16::in) >> (int::in) = (uint16::uo) is det.
:- func unchecked_right_shift(uint16::in, int::in) = (uint16::uo) is det.
:- func (uint16::in) /\ (uint16::in) = (uint16::uo) is det.
:- func (uint16::in) \/ (uint16::in) = (uint16::uo) is det.
:- func xor(uint16, uint16) = uint16.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (uint16::in) = (uint16::uo) is det.
:- func num_zeros(uint16) = int.
:- func num_ones(uint16) = int.
:- func num_leading_zeros(uint16) = int.
:- func num_trailing_zeros(uint16) = int.
:- func reverse_bytes(uint16) = uint16.
:- func reverse_bits(uint16) = uint16.
:- func rotate_left(uint16, uint) = uint16.
:- func unchecked_rotate_left(uint16, uint) = uint16.
:- func rotate_right(uint16, uint) = uint16.
:- func unchecked_rotate_right(uint16, uint) = uint16.
:- func set_bit(uint16, uint) = uint16.
:- func unchecked_set_bit(uint16, uint) = uint16.
:- func clear_bit(uint16, uint) = uint16.
:- func unchecked_clear_bit(uint16, uint) = uint16.
:- func flip_bit(uint16, uint) = uint16.
:- func unchecked_flip_bit(uint16, uint) = uint16.
:- pred bit_is_set(uint16::in, uint::in) is semidet.
:- pred unchecked_bit_is_set(uint16::in, uint::in) is semidet.
:- pred bit_is_clear(uint16::in, uint::in) is semidet.
:- pred unchecked_bit_is_clear(uint16::in, uint::in) is semidet.
:- func max_uint16 = uint16.
