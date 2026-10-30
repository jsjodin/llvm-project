% A declaration-only stub of the standard library module float, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module float.
:- interface.

:- func (float::in) + (float::in) = (float::uo) is det.
:- func (float::in) - (float::in) = (float::uo) is det.
:- func (float::in) * (float::in) = (float::uo) is det.
:- func (float::in) / (float::in) = (float::uo) is det.
:- func unchecked_quotient(float::in, float::in) = (float::uo) is det.
:- func + (float::in) = (float::uo) is det.
:- func - (float::in) = (float::uo) is det.
:- pred (float::in) < (float::in) is semidet.
:- pred (float::in) =< (float::in) is semidet.
:- pred (float::in) >= (float::in) is semidet.
:- pred (float::in) > (float::in) is semidet.
:- func float(int) = float.
:- func from_int8(int8) = float.
:- func from_int16(int16) = float.
:- func cast_from_int32(int32) = float.
:- func cast_from_int64(int64) = float.
:- func from_uint8(uint8) =  float.
:- func from_uint16(uint16) = float.
:- func cast_from_uint32(uint32) = float.
:- func cast_from_uint64(uint64) = float.
:- func ceiling_to_int(float) = int.
:- func floor_to_int(float) = int.
:- func round_to_int(float) = int.
:- func truncate_to_int(float) = int.
:- func abs(float) = float.
:- func max(float, float) = float.
:- func min(float, float) = float.
:- func pow(float, int) = float.
:- func hash(float) = int.
:- pred hash(float::in, int::out) is det.
:- pred is_infinite(float::in) is semidet.
:- pred is_inf(float::in) is semidet.
:- pred is_nan(float::in) is semidet.
:- pred is_nan_or_infinite(float::in) is semidet.
:- pred is_nan_or_inf(float::in) is semidet.
:- pred is_finite(float::in) is semidet.
:- pred is_zero(float::in) is semidet.
:- func max = float.
:- func min = float.
:- func infinity = float.
:- func epsilon = float.
:- func radix = int.
:- func mantissa_digits = int.
:- func min_exponent = int.
:- func max_exponent = int.
