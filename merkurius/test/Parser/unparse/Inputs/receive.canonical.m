% Receive goals laid out carelessly, for -freceive-goal; their canonical
% layout is receive.canonical.m.
:- module receive_messy.
:- implementation.
loop(Count, Self, Downs, !IO) :-
    receive [timeout(after(5000)), io(!IO)] (
        select(Self, increment(N)) ->
            loop(Count + N, Self, Downs, !IO)
    ;
        select(Self, get(ReplyTo)) ->
            send(ReplyTo, Count, !IO),
            loop(Count, Self, Downs, !IO)
    ;
        select(Self, stop) ->
            true
    ;
        select(Downs, _) ->
            % a comment in an arm
            io.write_string("worker died\n", !IO),
            loop(Count, Self, Downs, !IO)
    ;
        timeout ->
            ( if Count > 0 then
                io.write_string("idle\n", !IO)
            else
                true
            ),
            loop(Count, Self, Downs, !IO)
    ).
one(B, !IO) :-
    receive [io(!IO)] (
        select(B, X) ->
            write(X, !IO)
    ),
    nl(!IO).
nested(A, B, !IO) :-
    receive [io(!IO)] (
        select(A, reply(R, S)) ->
            receive [io(!IO)] (
                select(R, Y) ->
                    q(Y, S, !IO)
            ;
                select(B, _) ->
                    true
            )
    ;
        select(B, _) ->
            true
    ).
call(B, !IO) :-
    receive(B, any, infinity, _, !IO),
    process.receive(B, any, after(0), _, !IO).
