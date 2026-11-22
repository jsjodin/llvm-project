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
:- import_module io, process.

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

    % io.write and string.string for a term with unbound variables in it:
    % each is written as _1, _2 and so on, numbered in the order the write
    % meets them, the same variable with the same number, and a cell on
    % its own path, as a cyclic term has, as .... Impure, as var/1 is: what
    % is written changes when a variable is bound. A ground term is written
    % as io.write writes it.
:- impure pred write(T::ia, io::di, io::uo) is det.
:- impure func string(T::ia) = (string::out) is det.

    % process.send and process.receive for a message with unbound variables
    % in it. The message is copied as copy_term copies a term: bound
    % variables as their values, and each unbound variable as one new
    % variable, however often the message holds it, which the receiver can
    % bind and the sender's bindings no longer reach. send is impure, as
    % what it sends changes when a variable is bound. process.receive gives
    % a ground message, so a message whose type has open terms in it is
    % received with this receive, which takes the oldest.
:- impure pred send(mailbox(M)::in, M::ia, io::di, io::uo) is det.
:- pred receive(mailbox(M)::in, timeout::in, received(M)::oa,
    io::di, io::uo) is det.

%---------------------------------------------------------------------------%

:- implementation.

:- pragma foreign_decl("C", local, "
#include <stdint.h>
extern uint64_t merk_open_to_open(uint64_t, uint64_t);
extern int32_t merk_open_from_open(uint64_t, uint64_t, uint64_t *);
extern int32_t merk_open_unify_oc(uint64_t, uint64_t, uint64_t);
extern int32_t merk_open_is_var(uint64_t);
extern void merk_write(uint64_t, uint64_t, uint64_t);
extern MR_String merk_string_of(uint64_t, uint64_t);
").

:- pragma foreign_proc("C",
    write(X::ia, _IO0::di, _IO::uo),
    [will_not_call_merkurius, thread_safe],
"
    merk_write(TypeInfo_for_T, X, 0);
").

:- pragma foreign_proc("C",
    string(X::ia) = (S::out),
    [will_not_call_merkurius, thread_safe],
"
    S = merk_string_of(TypeInfo_for_T, X);
").

send(Mailbox, Message, !IO) :-
    impure process.send_any(Mailbox, Message, !IO).

    % process.receive's message is a new copy, which no one else holds:
    % ground to process.receive, which does not look into it, and any here.
receive(Mailbox, Timeout, Received, !IO) :-
    process.receive(Mailbox, process.any, Timeout, Received, !IO).

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
