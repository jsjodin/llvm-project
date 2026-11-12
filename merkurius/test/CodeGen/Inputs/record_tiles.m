:- module record_tiles.
:- interface.
:- import_module array.

    % A record, inline in arrays in every module.
:- type tile ---> tile(x :: int, live :: int, hue :: colour).
:- type colour ---> red ; blue.
    % Exported abstractly: inline in no module.
:- type opaque.
    % An enumeration exported abstractly, which no module can make a record's
    % field.
:- type shade.
    % Not a record in any module, by its shade.
:- type shaded ---> shaded(int, shade).

:- func tiles(int) = array(tile).
:- mode tiles(in) = array_uo is det.
:- func live_sum(array(tile)) = int.
:- pred bump(int::in, array(tile)::array_di, array(tile)::array_uo) is det.
:- func opaques(int) = array(opaque).
:- mode opaques(in) = array_uo is det.
:- func opaque_sum(array(opaque)) = int.
:- func shadeds(int) = array(shaded).
:- mode shadeds(in) = array_uo is det.
:- func dark_shade = shade.

:- implementation.
:- import_module int.

:- type opaque ---> opaque(int, int).
:- type shade ---> light ; dark.

tiles(N) = array.generate(N, tile_at).

:- func tile_at(int) = tile.
tile_at(I) = tile(I, I * I, ( if I mod 2 = 0 then red else blue )).

live_sum(A) = array.foldl(add_live, A, 0).

:- func add_live(tile, int) = int.
add_live(T, S) = S + T ^ live.

bump(I, !A) :-
    T = !.A ^ elem(I),
    !A ^ elem(I) := (T ^ live := T ^ live + 100).

opaques(N) = array.generate(N, (func(I) = opaque(I, -I))).

opaque_sum(A) = array.foldl((func(opaque(X, Y), S) = S + X - Y), A, 0).

shadeds(N) = array.generate(N, (func(I) = shaded(I, light))).

dark_shade = dark.
