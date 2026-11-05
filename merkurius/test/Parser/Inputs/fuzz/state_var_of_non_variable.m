% Each of these crashed the desugarer when !, !. and !: of a non-variable became
% constructors in data-terms, since the rules for quantifiers, trace and try
% parameters and !S ^ F := V let them through.
:- module state_var_of_non_variable.
:- implementation.
a(!IO) :- some [!f] true.
b(!IO) :- all [!.f] true.
c(!IO) :- trace [io(!f)] true.
d(!IO) :- trace [state(f, !f)] true.
e(!IO) :- try [io(!f)] true then true.
g(!R) :- !f ^ x := 1.
h(!R) :- require_switch_arms_det [!f] true.
