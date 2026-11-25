% Closures of polymorphic procedures built into code generation (design
% document, Lowering, "Closures"): the closure captures a type_info for each
% of the procedure's type variables, then the curried arguments, and the
% entry's body unifies or compares by those type_infos. The output is mmc's
% for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module builtin_closures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

% MERK: merk.closure @merk.builtin.builtin.unify_2_0$closure1(%{{.*}}, %{{.*}}) captures ["type_info", array<i64: 0, 0>]
% MERK: merk.proc private @merk.builtin.builtin.unify_2_0$closure1(%[[C:.*]]: !merk.closure, %{{.*}}: !merk.word)
% MERK-NEXT: merk.closure_word %[[C]], 0
% MERK: merk.closure_word %[[C]], 1

main(!IO) :-
    P = unify(1),
    ( if P(1) then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ),
    ( if P(2) then io.write_string("yes\n", !IO) else io.write_string("no\n", !IO) ),
    ( if list.all_true(unify("a"), ["a", "a"]) then
        io.write_string("all\n", !IO)
    else
        io.write_string("not all\n", !IO)
    ),
    Q = unify([1, 2]),
    ( if Q([1, 3]) then io.write_string("eq\n", !IO) else io.write_string("ne\n", !IO) ),
    L = (@<),
    ( if L("a", "b") then io.write_string("lt\n", !IO) else io.write_string("ge\n", !IO) ),
    list.filter('@<'(2), [1, 2, 3, 4], Big),
    io.write(Big, !IO),
    io.nl(!IO).

% OUTPUT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:all
% OUTPUT-NEXT:ne
% OUTPUT-NEXT:lt
% OUTPUT-NEXT:[3, 4]
