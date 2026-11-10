% Process declarations, :- process Kind receives Types, as the converter
% reads them with -fprocess-declarations (design document, "Processes and
% mailboxes"): each becomes a ProcessDecl with the kind's name, its argument
% types, its message types in order, and whether it runs on a GPU. The
% predicates it expands to are not in the dump, which shows the tree as
% written (Semantics/desugar/processes.m shows them). process, receives and
% on are operators only in an item that begins with :- process, so the
% module process, a constant on and a goal process read as before. Without
% -fprocess-declarations, none is an operator, as in mmc, and the
% declarations are syntax errors.
%
% RUN: merkc -fc1 -fprocess-declarations -fdebug-dump-parse-tree-no-sema %s \
% RUN:   | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --check-prefix=PLAIN
% PLAIN: processes.m:{{[0-9]+}}:{{[0-9]+}}: error: expected an operator or '.', but found 'counter'

:- module processes.
:- interface.
:- import_module io, list, process.

:- process counter(int) receives counter_msg.
% CHECK:      ProcessDecl
% CHECK-NEXT: | | | name: counter
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName int
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | receives:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName counter_msg
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | onGpu: false

    % No arguments, and a polymorphic message type.
:- process logger receives list(T).
% CHECK:      ProcessDecl
% CHECK-NEXT: | | | name: logger
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | receives:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName list
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | | | TypeVar T
% CHECK-NEXT: | | | onGpu: false

    % Several message types, a qualified type and a higher-order one.
:- process server(io.res(int), string) receives control_msg,
    (pred(int)), {int, data_msg}.
% CHECK:      ProcessDecl
% CHECK-NEXT: | | | name: server
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName io.res
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | | | NamedType
% CHECK-NEXT: | | | | | | | name: SymName int
% CHECK-NEXT: | | | | | | | arguments:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName string
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | receives:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName control_msg
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | HigherOrderType
% CHECK:      | | | | TupleType
% CHECK:      | | | onGpu: false

    % On a GPU, with parentheses.
:- process (summer(int) receives sum_msg) on gpu.
% CHECK:      ProcessDecl
% CHECK-NEXT: | | | name: summer
% CHECK:      | | | onGpu: true
:- process(adder(int) receives sum_msg on gpu).
% CHECK:      ProcessDecl
% CHECK-NEXT: | | | name: adder
% CHECK:      | | | onGpu: true

:- implementation.

:- type turn ---> on ; off.
% CHECK:      | | | | | name: SymName on

p(X) :- process, X = on, process.spawn(q, _, !IO).
% CHECK:      | | | | | Call
% CHECK-NEXT: | | | | | | name: SymName process
% CHECK:      | | | | | | right: FunctorData
% CHECK-NEXT: | | | | | | | name: SymName on
% CHECK:      | | | | | Call
% CHECK-NEXT: | | | | | | name: SymName process.spawn
