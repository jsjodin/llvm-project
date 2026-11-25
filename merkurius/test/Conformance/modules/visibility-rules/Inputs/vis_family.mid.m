:- module vis_family.mid.
:- interface.
:- include_module leaf.
:- func doubled(int) = int.
:- implementation.
:- import_module int.
doubled(N) = 2 * N.
:- func tripled(int) = int.
tripled(N) = 3 * N.
