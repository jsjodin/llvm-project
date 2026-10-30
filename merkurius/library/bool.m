% A declaration-only stub of the standard library module bool, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module bool.
:- interface.

:- type bool
    --->    no
    ;       yes.
:- func not(bool) = bool.
:- pred not(bool::in, bool::out) is det.
:- func or(bool, bool) = bool.
:- pred or(bool::in, bool::in, bool::out) is det.
:- func xor(bool, bool) = bool.
:- func and(bool, bool) = bool.
:- pred and(bool::in, bool::in, bool::out) is det.
:- func or_list(list(bool)) = bool.
:- pred or_list(list(bool)::in, bool::out) is det.
:- func and_list(list(bool)) = bool.
:- pred and_list(list(bool)::in, bool::out) is det.
:- func pred_to_bool((pred)::((pred) is semidet)) = (bool::out) is det.
