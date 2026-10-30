% A declaration-only stub of the standard library module uint32, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module uint32.
:- interface.
:- import_module int32.
:- import_module uint16.
:- import_module uint64.
:- import_module uint8.

:- pred from_int(int::in, uint32::out) is semidet.
:- func det_from_int(int) = uint32.
:- func cast_from_int(int) = uint32.
:- pred from_uint(uint::in, uint32::out) is semidet.
:- func det_from_uint(uint) = uint32.
:- func cast_from_uint(uint) = uint32.
:- func cast_to_int(uint32) = int.
:- func cast_to_uint(uint32) = uint.
:- func cast_to_uint8(uint32) = uint8.
:- func cast_from_uint8(uint8) = uint32.
:- func cast_to_uint16(uint32) = uint16.
:- func cast_from_uint16(uint16) = uint32.
:- func cast_to_uint64(uint32) = uint64.
:- func cast_from_uint64(uint64) = uint32.
:- func cast_from_int32(int32) = uint32.
:- func from_bytes_le(uint8, uint8, uint8, uint8) = uint32.
:- func from_bytes_be(uint8, uint8, uint8, uint8) = uint32.
:- pred (uint32::in) < (uint32::in) is semidet.
:- pred (uint32::in) > (uint32::in) is semidet.
:- pred (uint32::in) =< (uint32::in) is semidet.
:- pred (uint32::in) >= (uint32::in) is semidet.
:- func max(uint32, uint32) = uint32.
:- func min(uint32, uint32) = uint32.
:- func uint32 + uint32 = uint32.
:- mode in   + in  = uo is det.
:- mode uo   + in  = in is det.
:- mode in   + uo  = in is det.
:- func plus(uint32, uint32) = uint32.
:- func uint32 - uint32 = uint32.
:- mode in   - in   = uo is det.
:- mode uo   - in   = in is det.
:- mode in   - uo   = in is det.
:- func minus(uint32, uint32) = uint32.
:- func (uint32::in) * (uint32::in) = (uint32::uo) is det.
:- func times(uint32, uint32) = uint32.
:- func (uint32::in) div (uint32::in) = (uint32::uo) is det.
:- func (uint32::in) // (uint32::in) = (uint32::uo) is det.
:- func (uint32::in) / (uint32::in) = (uint32::uo) is det.
:- func unchecked_quotient(uint32::in, uint32::in) = (uint32::uo) is det.
:- func (uint32::in) mod (uint32::in) = (uint32::uo) is det.
:- func (uint32::in) rem (uint32::in) = (uint32::uo) is det.
:- func unchecked_rem(uint32::in, uint32::in) = (uint32::uo) is det.
:- pred even(uint32::in) is semidet.
:- pred odd(uint32::in) is semidet.
:- func (uint32::in) << (int::in) = (uint32::uo) is det.
:- func unchecked_left_shift(uint32::in, int::in) = (uint32::uo) is det.
:- func (uint32::in) >> (int::in) = (uint32::uo) is det.
:- func unchecked_right_shift(uint32::in, int::in) = (uint32::uo) is det.
:- func (uint32::in) /\ (uint32::in) = (uint32::uo) is det.
:- func (uint32::in) \/ (uint32::in) = (uint32::uo) is det.
:- func xor(uint32, uint32) = uint32.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (uint32::in) = (uint32::uo) is det.
:- func num_zeros(uint32) = int.
:- func num_ones(uint32) = int.
:- func num_leading_zeros(uint32) = int.
:- func num_trailing_zeros(uint32) = int.
:- func reverse_bytes(uint32) = uint32.
:- func reverse_bits(uint32) = uint32.
:- func rotate_left(uint32, uint) = uint32.
:- func unchecked_rotate_left(uint32, uint) = uint32.
:- func rotate_right(uint32, uint) = uint32.
:- func unchecked_rotate_right(uint32, uint) = uint32.
:- func set_bit(uint32, uint) = uint32.
:- func unchecked_set_bit(uint32, uint) = uint32.
:- func clear_bit(uint32, uint) = uint32.
:- func unchecked_clear_bit(uint32, uint) = uint32.
:- func flip_bit(uint32, uint) = uint32.
:- func unchecked_flip_bit(uint32, uint) = uint32.
:- pred bit_is_set(uint32::in, uint::in) is semidet.
:- pred unchecked_bit_is_set(uint32::in, uint::in) is semidet.
:- pred bit_is_clear(uint32::in, uint::in) is semidet.
:- pred unchecked_bit_is_clear(uint32::in, uint::in) is semidet.
:- func max_uint32 = uint32.
