% A declaration-only stub of the standard library module uint8, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module uint8.
:- interface.
:- import_module int8.

:- pred from_int(int::in, uint8::out) is semidet.
:- func det_from_int(int) = uint8.
:- func cast_from_int(int) = uint8.
:- pred from_uint(uint::in, uint8::out) is semidet.
:- func det_from_uint(uint) = uint8.
:- func cast_from_uint(uint) = uint8.
:- func to_int(uint8) = int.
:- func cast_to_int(uint8) = int.
:- func cast_to_uint(uint8) = uint.
:- func cast_from_int8(int8) = uint8.
:- pred (uint8::in) < (uint8::in) is semidet.
:- pred (uint8::in) > (uint8::in) is semidet.
:- pred (uint8::in) =< (uint8::in) is semidet.
:- pred (uint8::in) >= (uint8::in) is semidet.
:- func max(uint8, uint8) = uint8.
:- func min(uint8, uint8) = uint8.
:- func uint8 + uint8 = uint8.
:- mode in + in = uo is det.
:- mode uo + in = in is det.
:- mode in + uo = in is det.
:- func plus(uint8, uint8) = uint8.
:- func uint8 - uint8 = uint8.
:- mode in - in = uo is det.
:- mode uo - in = in is det.
:- mode in - uo = in is det.
:- func minus(uint8, uint8) = uint8.
:- func (uint8::in) * (uint8::in) = (uint8::uo) is det.
:- func times(uint8, uint8) = uint8.
:- func (uint8::in) div (uint8::in) = (uint8::uo) is det.
:- func (uint8::in) // (uint8::in) = (uint8::uo) is det.
:- func (uint8::in) / (uint8::in) = (uint8::uo) is det.
:- func unchecked_quotient(uint8::in, uint8::in) = (uint8::uo) is det.
:- func (uint8::in) mod (uint8::in) = (uint8::uo) is det.
:- func (uint8::in) rem (uint8::in) = (uint8::uo) is det.
:- func unchecked_rem(uint8::in, uint8::in) = (uint8::uo) is det.
:- pred even(uint8::in) is semidet.
:- pred odd(uint8::in) is semidet.
:- func (uint8::in) << (int::in) = (uint8::uo) is det.
:- func unchecked_left_shift(uint8::in, int::in) = (uint8::uo) is det.
:- func (uint8::in) >> (int::in) = (uint8::uo) is det.
:- func unchecked_right_shift(uint8::in, int::in) = (uint8::uo) is det.
:- func (uint8::in) /\ (uint8::in) = (uint8::uo) is det.
:- func (uint8::in) \/ (uint8::in) = (uint8::uo) is det.
:- func xor(uint8, uint8) = uint8.
:- mode xor(in, in) = uo is det.
:- mode xor(in, uo) = in is det.
:- mode xor(uo, in) = in is det.
:- func \ (uint8::in) = (uint8::uo) is det.
:- func num_zeros(uint8) = int.
:- func num_ones(uint8) = int.
:- func num_leading_zeros(uint8) = int.
:- func num_trailing_zeros(uint8) = int.
:- func reverse_bits(uint8) = uint8.
:- func rotate_left(uint8, uint) = uint8.
:- func unchecked_rotate_left(uint8, uint) = uint8.
:- func rotate_right(uint8, uint) = uint8.
:- func unchecked_rotate_right(uint8, uint) = uint8.
:- func set_bit(uint8, uint) = uint8.
:- func unchecked_set_bit(uint8, uint) = uint8.
:- func clear_bit(uint8, uint) = uint8.
:- func unchecked_clear_bit(uint8, uint) = uint8.
:- func flip_bit(uint8, uint) = uint8.
:- func unchecked_flip_bit(uint8, uint) = uint8.
:- pred bit_is_set(uint8::in, uint::in) is semidet.
:- pred unchecked_bit_is_set(uint8::in, uint::in) is semidet.
:- pred bit_is_clear(uint8::in, uint::in) is semidet.
:- pred unchecked_bit_is_clear(uint8::in, uint::in) is semidet.
:- func max_uint8 = uint8.
