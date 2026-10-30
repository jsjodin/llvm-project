% Field selection and update become calls to the field access functions
% (reference manual, "Record syntax"), and a function's head may be
% written in field access syntax. The module compiles with mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module fields.
:- interface.
:- import_module map, string.

:- type type1
        --->    type1(
                    field1 :: type2,
                    field2 :: string
                ).
:- type type2
        --->    type2(
                    field3 :: int,
                    field4 :: int
                ).

:- func type1 ^ increment_field3 = type1.
:- func update_field_in_map(map(int, type1), int, string)
    = map(int, type1) is semidet.
:- func type1 ^ first = type2.
:- func (type1 ^ first := type2) = type1.
:- pred bump(type2::in, type2::out) is det.
:- pred count(type2::in, type2::out) is det.

:- implementation.
:- import_module int.

% The manual's example ("Field access examples"), which a head in field
% access syntax defines.
Term0 ^ increment_field3 =
    Term0 ^ field1 ^ field3 := Term0 ^ field1 ^ field3 + 1.
% CHECK:      Function increment_field3(Term0) = $1
% CHECK-NEXT: | Conj #7
% CHECK-NEXT: | | Unify #0 $2 = field1(Term0)
% CHECK-NEXT: | | Unify #5 $3 = 'field3 :='($2, $4)
% CHECK-NEXT: | | Unify #4 $4 = '+'($5, $7)
% CHECK-NEXT: | | Unify #2 $5 = field3($6)
% CHECK-NEXT: | | Unify #1 $6 = field1(Term0)
% CHECK-NEXT: | | Unify #3 $7 = 1
% CHECK-NEXT: | | Unify #6 $1 = 'field1 :='(Term0, $3)

% With more fields, the old value of the first is updated in turn.
update_field_in_map(Map, Index, Value) =
    Map ^ elem(Index) ^ field2 := Value.
% CHECK:      Function update_field_in_map(Map, Index, Value) = $3
% CHECK-NEXT: | Conj #3
% CHECK-NEXT: | | Unify #0 $4 = elem(Index, Map)
% CHECK-NEXT: | | Unify #1 $5 = 'field2 :='($4, Value)
% CHECK-NEXT: | | Unify #2 $3 = 'elem :='(Index, Map, $5)

% Field access functions written by hand.
type1(Field1, _) ^ first = Field1.
% CHECK:      Function first($0) = Field1
% CHECK-NEXT: | Unify #0 $0 = type1(Field1, _$2)
(type1(_, Field2) ^ first := Field1) = type1(Field1, Field2).
% CHECK:      Function 'first :='($0, Field1) = $4
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #0 $0 = type1(_$1, Field2)
% CHECK-NEXT: | | Unify #1 $4 = type1(Field1, Field2)

% !S ^ Fields := V is !:S = !.S ^ Fields := V.
bump(!T) :-
    !T ^ field3 := !.T ^ field4.
% CHECK:      Predicate bump(T!0, T!1)
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #1 T!1 = 'field3 :='(T!0, $2)
% CHECK-NEXT: | | Unify #0 $2 = field4(T!0)

% The DCG forms: N =^ F selects from the DCG argument, and ^ F := V
% updates it.
count -->
    N =^ field3,
    ^ field3 := N + 1.
% CHECK:      Predicate count(dcg!0, dcg!1)
% CHECK-NEXT: | Conj #4
% CHECK-NEXT: | | Unify #0 N = field3(dcg!0)
% CHECK-NEXT: | | Unify #3 dcg!1 = 'field3 :='(dcg!0, $3)
% CHECK-NEXT: | | Unify #2 $3 = '+'(N, $4)
% CHECK-NEXT: | | Unify #1 $4 = 1
