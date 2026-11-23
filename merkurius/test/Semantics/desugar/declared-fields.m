% A module may declare the field access functions of its own fields
% (reference manual, "User-supplied field access function declarations"),
% to give them modes or export them. Field syntax then calls the declared
% function, and a declared one without clauses has the generated clause.
% The module compiles with mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s
% RUN: merkc -fc1 %s

:- module declared_fields.
:- interface.

:- type shape
    --->    rect(width :: int, height :: int)
    ;       square(side :: int).

:- func shape ^ width = int.
:- mode in ^ width = out is semidet.

:- func (shape ^ height := int) = shape.
:- mode (in ^ height := in) = out is semidet.

:- func area(shape) = int is semidet.

:- implementation.
:- import_module int.

area(S) = S ^ width * S ^ height.
% CHECK:      Function area(S) = $1
% CHECK:      Unify #{{[0-9]+}} $2 = width(S)

% CHECK:      Function width($0) = $1
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #0 $0 = rect($2, $3)
% CHECK-NEXT: | | Unify #1 $1 = $2
% CHECK:      Function 'height :='($0, $1) = $2
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #0 $0 = rect($3, $4)
% CHECK-NEXT: | | Unify #1 $2 = rect($3, $1)
