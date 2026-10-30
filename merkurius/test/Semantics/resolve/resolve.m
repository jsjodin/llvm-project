% Name resolution: each name is resolved to every declaration it can refer
% to, by its use and arity, among the module's own declarations and those
% of the modules it imports. A data-term f(X1, ..., Xn) can be a
% constructor, a field, a function of arity n, or a closure of a predicate
% or function with more arguments; an overloaded name keeps all of them,
% as mmc lists them. A clause without a declaration declares a local
% predicate or function, whose type is to be inferred (mmc --infer-all).
%
% RUN: merkc -fc1 -fdebug-dump-resolved-names %s 2>&1 | FileCheck --implicit-check-not=error %s

:- module resolve.
:- interface.
:- import_module list.
:- type tree(T) ---> leaf ; node(left :: tree(T), val :: T, right :: tree(T)).
:- pred size(tree(T)::in, int::out) is det.
:- func sum(list(int)) = int.
:- pred show(int::in, string::out) is det.
:- implementation.
:- import_module int.
:- use_module string.

size(leaf, 0).
size(node(L, _, R), N) :- size(L, A), size(R, B), N = A + B + 1.
sum(Xs) = foldl(add, Xs, 0).
add(X, Y) = X + Y.
show(X, S) :- S = string.from_int(X).

% CHECK:      symbols of resolve:
% CHECK-NEXT:   14:9 type resolve.tree/1 exported
% CHECK-NEXT:   14:22 constructor resolve.leaf/0 exported of type resolve.tree/1
% CHECK-NEXT:   14:29 constructor resolve.node/3 exported of type resolve.tree/1
% CHECK-NEXT:   14:34 field resolve.left/1 exported of type resolve.tree/1
% CHECK-NEXT:   14:34 field update resolve.'left :='/2 exported of type resolve.tree/1
% CHECK-NEXT:   14:51 field resolve.val/1 exported of type resolve.tree/1
% CHECK-NEXT:   14:51 field update resolve.'val :='/2 exported of type resolve.tree/1
% CHECK-NEXT:   14:61 field resolve.right/1 exported of type resolve.tree/1
% CHECK-NEXT:   14:61 field update resolve.'right :='/2 exported of type resolve.tree/1
% CHECK-NEXT:   15:9 pred resolve.size/2 exported
% CHECK-NEXT:   16:9 func resolve.sum/1 exported
% CHECK-NEXT:   17:9 pred resolve.show/2 exported
% CHECK-NEXT:   25:1 func resolve.add/2
% CHECK-NEXT: names:
% CHECK-NEXT:   14:42 tree Type/1 -> type resolve.tree/1
% CHECK-NEXT:   14:70 tree Type/1 -> type resolve.tree/1
% CHECK-NEXT:   15:14 tree Type/1 -> type resolve.tree/1
% CHECK-NEXT:   15:23 in Mode/0 -> mode builtin.in/0
% CHECK-NEXT:   15:27 int Type/0 -> type int/0
% CHECK-NEXT:   15:32 out Mode/0 -> mode builtin.out/0
% CHECK-NEXT:   16:18 int Type/0 -> type int/0
% CHECK-NEXT:   16:13 list Type/1 -> type list.list/1
% CHECK-NEXT:   16:26 int Type/0 -> type int/0
% CHECK:        22:1 size Definition/2 -> pred resolve.size/2
% CHECK-NEXT:   22:6 leaf DataTerm/0 -> constructor resolve.leaf/0
% CHECK-NEXT:   23:1 size Definition/2 -> pred resolve.size/2
% CHECK-NEXT:   23:6 node DataTerm/3 -> constructor resolve.node/3
% CHECK-NEXT:   23:27 size Call/2 -> pred resolve.size/2
% CHECK-NEXT:   23:39 size Call/2 -> pred resolve.size/2
% CHECK-NEXT:   23:61 + DataTerm/2 -> func int.'+'/2
% CHECK-NEXT:   23:57 + DataTerm/2 -> func int.'+'/2
% CHECK-NEXT:   24:1 sum Definition/1 -> func resolve.sum/1
% CHECK-NEXT:   24:11 foldl DataTerm/3 -> func list.foldl/3, pred list.foldl/4
% CHECK-NEXT:   24:17 add DataTerm/0 -> func resolve.add/2
% CHECK-NEXT:   25:1 add Definition/2 -> func resolve.add/2
% CHECK-NEXT:   25:15 + DataTerm/2 -> func int.'+'/2
% CHECK-NEXT:   26:1 show Definition/2 -> pred resolve.show/2
% CHECK-NEXT:   26:19 string.from_int DataTerm/1 -> func string.from_int/1
