:- module vis_family.mid.leaf.
:- interface.
:- func peek = int.
:- func bump(counter) = counter.
:- func wrapped = secret.
:- implementation.
:- import_module int.
peek = tripled(doubled(reveal(secret(list.length([1, 2, 3]))))).
bump(counter(N)) = counter(N + 1).
wrapped = secret(5).
