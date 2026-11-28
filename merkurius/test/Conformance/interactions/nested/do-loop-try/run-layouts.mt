% MANUAL: Discriminated unions
% DESIGN: Loops
% MANUAL: Exception handling
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in values
% thrown as exceptions and caught, in a do loop that makes, gathers and
% carries the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_do_loop_try_layouts_translation.mt
% MODULES: Inputs/nested_do_loop_try_layouts_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_do_loop_try_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_do_loop_try_layouts_float.m %t.float.d/nested_do_loop_try_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_do_loop_try_layouts_float.m %t.float.d/nested_do_loop_try_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) / returned seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) / caught seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441)) / returned seven(three(10.5, one(21), 63), four(x, 441))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / caught seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / returned seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0)) / caught seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / returned seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) / caught seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23)) / returned seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / caught seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) / returned seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0)) / caught seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_do_loop_try_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_do_loop_try_layouts_float32.m %t.float32.d/nested_do_loop_try_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_do_loop_try_layouts_float32.m %t.float32.d/nested_do_loop_try_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) / returned seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) / caught seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441)) / returned seven(three(10.5, one(21), 63), four(x, 441))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / caught seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / returned seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0)) / caught seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / returned seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) / caught seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23)) / returned seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / caught seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) / returned seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0)) / caught seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_do_loop_try_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_do_loop_try_layouts_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.
:- func run(int) = string.

run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make_nested(K),
        Ss1 = [use_nested(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make_nested(N), V0, V1, Last) do
        ( if I > 0 then V1 = make_nested(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use_nested(Last).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
