:- module field_types.
:- interface.
:- type account
    --->    account(
                owner :: string,
                balance :: int
            ).
:- type entry(T)
    --->    entry(key :: int, item :: T)
    ;       blank.
:- implementation.
