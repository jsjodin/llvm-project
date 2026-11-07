% The registry of mailbox names, as Erlang's register/2 and whereis/1, for
% which mmc cannot be the oracle, as it has no process module.
%
% - register's errors: a name that is taken, a mailbox that has a name,
%   and a closed mailbox.
% - lookup's: no mailbox of the name, and messages of another type, which
%   differ in a type argument only, or are looked up through a type_info
%   in polymorphic code.
% - A looked-up handle is the registered one, by unification, and a send
%   to it reaches the mailbox; two mailboxes' handles differ.
% - Closing a mailbox and unregister remove the name, which can be
%   registered again.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:register server: ok
% OUTPUT-NEXT:register server again: error("the name \"server\" is taken")
% OUTPUT-NEXT:register the server as other: error("the mailbox has a name already")
% OUTPUT-NEXT:register lists: ok
% OUTPUT-NEXT:the server is on this node: yes
% OUTPUT-NEXT:found the registered handle: yes
% OUTPUT-NEXT:two mailboxes' handles differ: yes
% OUTPUT-NEXT:reply: yes("pong")
% OUTPUT-NEXT:lists as itself: found
% OUTPUT-NEXT:lists as list(pair(string, int)): the mailbox registered as "lists" has messages of another type
% OUTPUT-NEXT:lists as list(int): the mailbox registered as "lists" has messages of another type
% OUTPUT-NEXT:lists through a type_info: found
% OUTPUT-NEXT:lists as maybe(int) through a type_info: the mailbox registered as "lists" has messages of another type
% OUTPUT-NEXT:nobody: no mailbox is registered as "nobody"
% OUTPUT-NEXT:server after its close: no mailbox is registered as "server"
% OUTPUT-NEXT:register the closed server: error("the mailbox is closed or another node\'s")
% OUTPUT-NEXT:lists after unregister: no mailbox is registered as "lists"
% OUTPUT-NEXT:register lists again: ok
% OUTPUT-NEXT:lists again: found
% OUTPUT-NEXT:register server for another mailbox: ok

:- module process_registry.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, list, maybe, pair, process, string.

:- type request
    --->    ping(mailbox(string))
    ;       stop.

:- type lists == list(pair(int, string)).

main(!IO) :-
    this_node(Self, !IO),
    new_mailbox(Server : mailbox(request), !IO),
    new_mailbox(Lists : mailbox(lists), !IO),
    register("server", Server, R1, !IO),
    show("register server", R1, !IO),
    register("server", Lists, R2, !IO),
    show("register server again", R2, !IO),
    register("other", Server, R3, !IO),
    show("register the server as other", R3, !IO),
    register("lists", Lists, R4, !IO),
    show("register lists", R4, !IO),
    yes_no("the server is on this node",
        ( if node_of(Server) = Self then yes else no ), !IO),

    lookup(Self, "server", after(0), Found : maybe_error(mailbox(request)),
        !IO),
    (
        Found = ok(S),
        yes_no("found the registered handle",
            ( if S = Server then yes else no ), !IO),
        new_mailbox(Reply, !IO),
        yes_no("two mailboxes' handles differ",
            ( if Reply = to_string_mailbox(Server) then no else yes ), !IO),
        send(S, ping(Reply), !IO),
        receive(Server, any, after(0), Got, !IO),
        ( if Got = yes(ping(R)) then
            send(R, "pong", !IO),
            receive(Reply, any, after(0), Answer, !IO),
            io.write_string("reply: ", !IO),
            io.write_line(Answer, !IO)
        else
            io.write_string("no ping\n", !IO)
        )
    ;
        Found = error(E),
        io.write_string(E ++ "\n", !IO)
    ),

    lookup(Self, "lists", after(0), L1 : maybe_error(mailbox(lists)), !IO),
    result("lists as itself", L1, !IO),
    lookup(Self, "lists", after(0),
        L2 : maybe_error(mailbox(list(pair(string, int)))), !IO),
    result("lists as list(pair(string, int))", L2, !IO),
    lookup(Self, "lists", after(0), L3 : maybe_error(mailbox(list(int))),
        !IO),
    result("lists as list(int)", L3, !IO),
    poly_lookup(Self, "lists", [1 - "a"], L4, !IO),
    result("lists through a type_info", L4, !IO),
    poly_lookup(Self, "lists", yes(1), L5, !IO),
    result("lists as maybe(int) through a type_info", L5, !IO),
    lookup(Self, "nobody", after(0), L6 : maybe_error(mailbox(int)), !IO),
    result("nobody", L6, !IO),

    close_mailbox(Server, !IO),
    lookup(Self, "server", after(0), L7 : maybe_error(mailbox(request)),
        !IO),
    result("server after its close", L7, !IO),
    register("again", Server, R5, !IO),
    show("register the closed server", R5, !IO),
    unregister("lists", !IO),
    lookup(Self, "lists", after(0), L8 : maybe_error(mailbox(lists)), !IO),
    result("lists after unregister", L8, !IO),
    register("lists", Lists, R6, !IO),
    show("register lists again", R6, !IO),
    lookup(Self, "lists", after(0), L9 : maybe_error(mailbox(lists)), !IO),
    result("lists again", L9, !IO),
    new_mailbox(Server2 : mailbox(request), !IO),
    register("server", Server2, R7, !IO),
    show("register server for another mailbox", R7, !IO).

    % Looks up a mailbox whose message type is the example's, which only a
    % type_info gives here.
:- pred poly_lookup(node::in, string::in, T::in,
    maybe_error(mailbox(T))::out, io::di, io::uo) is det.

poly_lookup(Node, Name, _, Result, !IO) :-
    lookup(Node, Name, after(0), Result, !IO).

    % A handle of another type with the same words, to compare two
    % mailboxes' handles.
:- func to_string_mailbox(mailbox(request)) = mailbox(string).
:- pragma foreign_proc("C",
    to_string_mailbox(M::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = M;
").

:- pred show(string::in, maybe_error::in, io::di, io::uo) is det.

show(What, Result, !IO) :-
    io.write_string(What ++ ": ", !IO),
    io.write_line(Result, !IO).

:- pred result(string::in, maybe_error(T)::in, io::di, io::uo) is det.

result(What, ok(_), !IO) :-
    io.write_string(What ++ ": found\n", !IO).
result(What, error(E), !IO) :-
    io.write_string(What ++ ": " ++ E ++ "\n", !IO).

:- pred yes_no(string::in, bool::in, io::di, io::uo) is det.

yes_no(What, Value, !IO) :-
    io.write_string(What ++ ": ", !IO),
    io.write_line(Value, !IO).
