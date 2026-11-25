% Writes terms as Merkurius' io.write and open.write write them, for the
% Prolog translations of the open-types tests: a list as [1, 2, 3], a
% compound term as f(A, B), a tuple '{}'(...) as {A, B}, and each unbound
% variable as _1, _2 and so on, numbered in the order the write meets
% them, depth first and left to right.

mercury_write(Term) :-
    \+ \+ ( term_variables(Term, Vars),
            name_variables(Vars, 1),
            write_term_m(Term) ).

mercury_write_line(Term) :-
    mercury_write(Term),
    nl.

name_variables([], _).
name_variables([V | Vs], N) :-
    atom_concat('_', N, V),
    N1 is N + 1,
    name_variables(Vs, N1).

write_term_m(T) :-
    is_list(T), !,
    write('['), write_arguments(T), write(']').
write_term_m([H | T]) :- !,
    % A partial list: [1, 2 | _1].
    write('['), write_term_m(H), write_tail(T), write(']').
write_term_m(T) :-
    compound(T), compound_name_arguments(T, '{}', [Args]), !,
    comma_list(Args, List),
    write('{'), write_arguments(List), write('}').
write_term_m(T) :-
    compound(T), !,
    compound_name_arguments(T, Name, Args),
    write(Name), write('('), write_arguments(Args), write(')').
write_term_m(T) :-
    write(T).

write_tail(T) :-
    is_list(T), !,
    ( T = [] -> true ; write(', '), write_arguments(T) ).
write_tail([H | T]) :- !,
    write(', '), write_term_m(H), write_tail(T).
write_tail(T) :-
    write(' | '), write_term_m(T).

write_arguments([]).
write_arguments([A]) :- !,
    write_term_m(A).
write_arguments([A | As]) :-
    write_term_m(A), write(', '), write_arguments(As).

comma_list((A, B), [A | Bs]) :- !,
    comma_list(B, Bs).
comma_list(A, [A]).

% from_open: the term, if no variable in it is unbound and it is not
% cyclic.
from_open(Open, Ground) :-
    acyclic_term(Open),
    ground(Open),
    Ground = Open.
