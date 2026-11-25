% Arrays in polymorphic code (design document, Lowering, "Arrays"): set,
% fill, resize and shrink of an array(T) whose elements' width only the
% type_info gives, for elements of a word, of the narrow widths, and of flat
% records, which a get boxes and a set unboxes. The output is mmc's for the
% same program.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -S %s -o - | FileCheck %s --check-prefix=ASM
% ASM-DAG: merk_array_fill_typed
% ASM-DAG: merk_array_resize_typed
% ASM-DAG: merk_array_load_record
% ASM-DAG: merk_array_store_record

:- module polymorphic_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string, char.

:- pred twiddle(T::in, T::in, array(T)::array_di, array(T)::array_uo) is det.
twiddle(X, Y, !A) :-
    array.set(0, X, !A),
    array.fill(Y, !A),
    array.set(1, X, !A),
    array.resize(5, X, !A),
    array.shrink(4, !A).

:- func get_all(array(T)) = list(T).
get_all(A) = L :- L = list.map(func(I) = array.lookup(A, I), 0 .. array.size(A) - 1).

:- pred show(array(T)::in, io::di, io::uo) is det.
show(A, !IO) :- io.write(get_all(A), !IO), io.nl(!IO).

:- type rec ---> rec(int, string).

main(!IO) :-
    twiddle(1, 2, array.init(3, 0), A1), show(A1, !IO),
    twiddle(1.5, 2.5, array.init(3, 0.0), A2), show(A2, !IO),
    twiddle("a", "b", array.init(3, ""), A3), show(A3, !IO),
    twiddle('x', 'y', array.init(3, 'z'), A4), show(A4, !IO),
    twiddle(rec(1, "r"), rec(2, "s"), array.init(3, rec(0, "")), A5), show(A5, !IO),
    twiddle(7u8, 9u8, array.init(3, 0u8), A6), show(A6, !IO),
    twiddle(7i16, -9i16, array.init(3, 0i16), A7), show(A7, !IO),
    twiddle(70000i32, -9i32, array.init(3, 0i32), A8), show(A8, !IO),
    twiddle([1], [], array.init(2, [3]), A9), show(A9, !IO).

% OUTPUT:[2, 1, 2, 1]
% OUTPUT-NEXT:[2.5, 1.5, 2.5, 1.5]
% OUTPUT-NEXT:["b", "a", "b", "a"]
% OUTPUT-NEXT:['y', 'x', 'y', 'x']
% OUTPUT-NEXT:[rec(2, "s"), rec(1, "r"), rec(2, "s"), rec(1, "r")]
% OUTPUT-NEXT:[9u8, 7u8, 9u8, 7u8]
% OUTPUT-NEXT:[-9i16, 7i16, -9i16, 7i16]
% OUTPUT-NEXT:[-9i32, 70000i32, -9i32, 70000i32]
% OUTPUT-NEXT:{{\[\[}}], [1], [1], [1]]
