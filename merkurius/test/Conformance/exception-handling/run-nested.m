% MANUAL: Exception handling
% Nested try goals with io parameters: an exception the inner one does not
% catch reaches the outer one.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:outer goal
% OUTPUT-NEXT:inner goal
% OUTPUT-NEXT:outer caught inner
% OUTPUT-NOT:{{.}}

:- module run_nested.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [io(!IO)] (
        io.write_string("outer goal\n", !IO),
        ( try [io(!IO)] (
            io.write_string("inner goal\n", !IO),
            throw("inner")
          )
          then io.write_string("inner then\n", !IO)
          catch 0 -> io.write_string("zero\n", !IO)
        ),
        io.write_string("not reached\n", !IO)
      )
    then
        io.write_string("outer then\n", !IO)
    catch S ->
        io.write_string("outer caught " ++ S ++ "\n", !IO)
    ).
