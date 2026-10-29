% The module system's declarations, as the converter reads them.
%
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERROR %s < %t.err

:- module merkurius.test__example.
:- interface.
:- import_module io, list.
:- implementation.
:- use_module (string), (map.tree).
:- include_module sub.
:- import_module "string".
:- end_module merkurius.test__example.

% CHECK:      ParseTree
% CHECK-NEXT: | items:
% CHECK-NEXT: | | ModuleDecl
% CHECK-NEXT: | | | name: SymName merkurius.test.example
% CHECK-NEXT: | | SectionDecl
% CHECK-NEXT: | | | section: Interface
% CHECK-NEXT: | | ImportDecl
% CHECK-NEXT: | | | kind: ImportModule
% CHECK-NEXT: | | | modules:
% CHECK-NEXT: | | | | SymName io
% CHECK-NEXT: | | | | SymName list
% CHECK-NEXT: | | SectionDecl
% CHECK-NEXT: | | | section: Implementation
% CHECK-NEXT: | | ImportDecl
% CHECK-NEXT: | | | kind: UseModule
% CHECK-NEXT: | | | modules:
% CHECK-NEXT: | | | | SymName string
% CHECK-NEXT: | | | | SymName map.tree
% CHECK-NEXT: | | ImportDecl
% CHECK-NEXT: | | | kind: IncludeModule
% CHECK-NEXT: | | | modules:
% CHECK-NEXT: | | | | SymName sub
% CHECK-NEXT: | | EndModuleDecl
% CHECK-NEXT: | | | name: SymName merkurius.test.example
% CHECK-EMPTY:

% ERROR: module-system.m:12:18: error: expected a module name, but found '"string"'
% ERROR-NOT: error:
