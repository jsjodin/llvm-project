% Open types, a Merkurius extension with no counterpart in the Mercury
% library, loaded only with -fopen-types (design document, Language
% extensions, "Open terms"). For any type T, open(T) has T's constructors,
% but any argument at any depth may be an unbound variable: an argument of
% type U has type open(U) inside it, but one of type closed(U), which holds a
% ground U. Values of an open type have the inst any, and are passed with the
% modes ia and oa. In a constructor's arguments, a ground U is accepted where
% open(U) is expected; elsewhere to_open converts.
%
% The predicates call the runtime's generic walkers (runtime/open.h), by
% T's type_info. Open terms are for the host only.

:- module open.
:- interface.

    % A T whose parts may be unbound variables. The type checker gives it
    % T's constructors.
:- type open(T).

    % A ground T inside an open type: open(closed(T)) holds a T as it is,
    % which is never walked, copied or bound into. Outside open(_) it is T's
    % word, at no cost.
:- type closed(T)
    --->    closed(T).

    % The ground term as an open one, shared where the two layouts are the
    % same, and copied where they differ.
:- func to_open(T) = open(T).

    % The open term as a ground one, if no variable in it is unbound and it
    % is not cyclic; shared if it holds no variable cell, bound or not.
    % Impure, as var/1 is: it can fail and then succeed once a hole is
    % bound, so a pure from_open could run before the goals that bind it.
:- impure pred from_open(open(T)::ia, T::out) is semidet.

    % Unification with the occurs check, where = has none.
:- pred unify_oc(open(T)::ia, open(T)::ia) is semidet.

    % Whether the term is an unbound variable, or is not: non-logical, so
    % impure, as the answer changes when the variable is bound.
:- impure pred var(open(T)::ia) is semidet.
:- impure pred nonvar(open(T)::ia) is semidet.

%---------------------------------------------------------------------------%

:- implementation.

:- pragma foreign_decl("C", local, "
#include <stdint.h>
extern uint64_t merk_open_to_open(uint64_t, uint64_t);
extern int32_t merk_open_from_open(uint64_t, uint64_t, uint64_t *);
extern int32_t merk_open_unify_oc(uint64_t, uint64_t, uint64_t);
extern int32_t merk_open_is_var(uint64_t);
").

:- pragma foreign_proc("C",
    to_open(X::in) = (Y::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Y = merk_open_to_open(TypeInfo_for_T, X);
").

:- pragma foreign_proc("C",
    from_open(X::ia, Y::out),
    [will_not_call_merkurius, thread_safe],
"
    uint64_t value = 0;
    SUCCESS_INDICATOR = merk_open_from_open(TypeInfo_for_T, X, &value);
    Y = (MR_Word) value;
").

:- pragma foreign_proc("C",
    unify_oc(X::ia, Y::ia),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = merk_open_unify_oc(TypeInfo_for_T, X, Y);
").

:- pragma foreign_proc("C",
    var(X::ia),
    [will_not_call_merkurius, thread_safe],
"
    SUCCESS_INDICATOR = merk_open_is_var(X);
").

:- pragma foreign_proc("C",
    nonvar(X::ia),
    [will_not_call_merkurius, thread_safe],
"
    SUCCESS_INDICATOR = !merk_open_is_var(X);
").
