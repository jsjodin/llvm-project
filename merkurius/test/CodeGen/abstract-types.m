% Values of abstract types of another module (Inputs/secret.m), whose
% constructors this module cannot see: a type of cells, an enum and a
% polymorphic type, passed to and from that module's procedures, stored in
% this module's terms and lists, tested and compared through their
% descriptors, which that module defines, and written. Such a type has no
% merk.type_def here; that module's says it is abstract (Lowering checks it
% passes as a pointer, abstract-type-errors.m). The output is mmc's for the
% same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu -I %S/Inputs %s | FileCheck %s --check-prefix=MERK --implicit-check-not='merk.type_def "secret.'
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %S/Inputs/secret.m | FileCheck %s --check-prefix=SECRET
% RUN: merkc -O0 -I %S/Inputs %S/Inputs/secret.m %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %S/Inputs/secret.m %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module abstract_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, secret.

:- type holder ---> holder(box(int), colour).
:- type wrapped ---> wrapped(colour).

main(!IO) :-
    B = box(42),
    io.write_int(unbox(B), !IO),
    io.nl(!IO),
    C = blue,
    io.write_string(name(C), !IO),
    io.nl(!IO),
    ( if C = red then io.write_string("same\n", !IO) else io.write_string("different\n", !IO) ),
    ( if blue = C then io.write_string("same\n", !IO) else io.write_string("different\n", !IO) ),
    compare(R, red, blue),
    io.write(R, !IO),
    io.nl(!IO),
    io.write(B, !IO),
    io.nl(!IO),
    H = holder(box(7), red),
    H = holder(B2, C2),
    io.write_int(unbox(B2), !IO),
    io.write_string(name(C2), !IO),
    io.nl(!IO),
    io.write(H, !IO),
    io.nl(!IO),
    W = wrapped(blue),
    W = wrapped(C3),
    io.write_string(name(C3), !IO),
    io.nl(!IO),
    L = [red, blue, red],
    io.write(L, !IO),
    io.nl(!IO),
    P0 = pair_of("a", "b"),
    swap(P0, P),
    show_pair(P, !IO),
    F = (func(X) = name(X)),
    io.write_string(F(blue), !IO),
    io.nl(!IO).

% MERK: merk.type_def "abstract_types.holder" [#merk.constructor<"holder"(!merk.term<"secret.box(int)">, !merk.term<"secret.colour">)>]
% MERK-LABEL: merk.proc @abstract_types.main_2_0()
% MERK: merk.call @secret.box_f1_0(%{{.*}}, %{{.*}}) : (!merk.type_info, !merk.word) -> !merk.term<"secret.box(T0)">
% MERK: merk.call @secret.blue_f0_0() : () -> !merk.term<"secret.colour">
% MERK: %[[INFO:.*]] = merk.type_info "secret.colour/0"
% MERK: merk.unify_word %[[INFO]], %{{.*}}, %{{.*}}
% MERK: merk.compare_word %{{.*}}, %{{.*}}, %{{.*}} : <"builtin.comparison_result">

% SECRET-DAG: merk.type_def "secret.box(T0)" [#merk.constructor<"box"(!merk.word, i64)>] abstract
% SECRET-DAG: merk.type_def "secret.colour" [#merk.constructor<"crimson">, #merk.constructor<"navy">, #merk.constructor<"green">] abstract
% SECRET-DAG: merk.type_def "secret.pair_of(T0)" [#merk.constructor<"two"(!merk.word, !merk.word)>] abstract

% OUTPUT:42
% OUTPUT-NEXT:blue
% OUTPUT-NEXT:different
% OUTPUT-NEXT:same
% OUTPUT-NEXT:'<'
% OUTPUT-NEXT:box(42, 0)
% OUTPUT-NEXT:7red
% OUTPUT-NEXT:holder(box(7, 0), crimson)
% OUTPUT-NEXT:blue
% OUTPUT-NEXT:[crimson, navy, crimson]
% OUTPUT-NEXT:two("b", "a")
% OUTPUT-NEXT:blue
% OUTPUT-EMPTY:
