% Open types, a Merkurius extension with no counterpart in the Mercury
% library, loaded only with -fopen-types (design document, Language
% extensions, "Open terms"). For any type T, open(T) has T's constructors,
% but any argument at any depth may be an unbound variable: an argument of
% type U has type open(U) inside it, but one of type closed(U), which holds a
% ground U. Values of an open type have the inst any, and are passed with the
% modes ia and oa. In a constructor's arguments, a ground U is accepted where
% open(U) is expected; elsewhere to_open converts.
%
% Temporary (open types step 1, docs/INPROGRESS.md): declarations only, for
% the type checker and mode analysis; steps 3 and 4 give them code.

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
:- pred from_open(open(T)::ia, T::out) is semidet.

    % Unification with the occurs check, where = has none.
:- pred unify_oc(open(T)::ia, open(T)::ia) is semidet.

    % Whether the term is an unbound variable, or is not: non-logical, so
    % impure, as the answer changes when the variable is bound.
:- impure pred var(open(T)::ia) is semidet.
:- impure pred nonvar(open(T)::ia) is semidet.
