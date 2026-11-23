% MANUAL: Subtypes
% The manual's subtypes: one that omits constructors, one that replaces an
% argument's type with a subtype, a polymorphic one of list, one of a
% higher-order constructor, and the mutually recursive bipartite graph.
% Terms convert to the supertype with coerce, and a subtype's values compare
% as its base type's.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_manual_examples.citrus_fruit

:- module nyi_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type fruit
    --->    apple
    ;       pear
    ;       lemon
    ;       orange.

:- type citrus_fruit =< fruit
    --->    lemon
    ;       orange.

:- type fruit_basket
    --->    basket(fruit, int).

:- type citrus_fruit_basket =< fruit_basket
    --->    basket(citrus_fruit, int).

:- type non_empty_list(T) =< list(T)
    --->    [T | list(T)].

:- type task
    --->    create(pred(int::in, io::di, io::uo) is det)
    ;       delete(pred(int::in, io::di, io::uo) is det).

:- type create_task =< task
    --->    create(pred(int::in, io::di, io::uo) is det).

:- type source_id == int.

:- type or_node
    --->    or_source(source_id)
    ;       logical_or(and_node, and_node)
    ;       logical_or_list(and_node, and_node, and_node, list(and_node)).

:- type and_node
    --->    and_source(source_id)
    ;       logical_and(or_node, or_node)
    ;       logical_and_list(or_node, or_node, or_node, list(or_node)).

:- type binary_or_node =< or_node
    --->    or_source(source_id)
    ;       logical_or(binary_or_and_node, binary_or_and_node).

:- type binary_or_and_node =< and_node
    --->    and_source(source_id)
    ;       logical_and(binary_or_node, binary_or_node)
    ;       logical_and_list(binary_or_node, binary_or_node, binary_or_node,
                list(binary_or_node)).

main(!IO) :-
    C = lemon : citrus_fruit,
    F = coerce(C) : fruit,
    io.write_line(F, !IO),
    B = basket(orange, 3) : citrus_fruit_basket,
    FB = coerce(B) : fruit_basket,
    io.write_line(FB, !IO),
    NE = [1, 2, 3] : non_empty_list(int),
    L = coerce(NE) : list(int),
    io.write_line(list.length(L) : int, !IO),
    io.write_line(head(NE), !IO),
    T = create(show) : create_task,
    run(coerce(T), !IO),
    G = logical_or(and_source(1),
        logical_and_list(or_source(2), or_source(3), or_source(4), []))
        : binary_or_node,
    io.write_line(coerce(G) : or_node, !IO),
    io.write_line(sources(coerce(G)), !IO),
    ( if compare((<), coerce(lemon : citrus_fruit) : fruit, orange) then
        io.write_string("lemon < orange\n", !IO)
    else
        io.write_string("lemon >= orange\n", !IO)
    ),
    ( if (lemon : citrus_fruit) = lemon then
        io.write_string("equal\n", !IO)
    else
        io.write_string("unequal\n", !IO)
    ).

:- func head(non_empty_list(T)) = T.
head([X | _]) = X.

:- pred show(int::in, io::di, io::uo) is det.
show(N, !IO) :-
    io.format("create %d\n", [i(N)], !IO).

:- pred run(task::in, io::di, io::uo) is det.
run(create(P), !IO) :- P(7, !IO).
run(delete(P), !IO) :- P(8, !IO).

:- func sources(or_node) = int.
sources(or_source(N)) = N.
sources(logical_or(A, B)) = and_sources(A) + and_sources(B).
sources(logical_or_list(A, B, C, Ds)) =
    and_sources(A) + and_sources(B) + and_sources(C) +
    list.foldl(func(D, S) = S + and_sources(D), Ds, 0).

:- func and_sources(and_node) = int.
and_sources(and_source(N)) = N.
and_sources(logical_and(A, B)) = sources(A) + sources(B).
and_sources(logical_and_list(A, B, C, Ds)) =
    sources(A) + sources(B) + sources(C) +
    list.foldl(func(D, S) = S + sources(D), Ds, 0).
