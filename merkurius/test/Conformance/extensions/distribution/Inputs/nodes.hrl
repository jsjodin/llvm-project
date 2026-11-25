%% Starts and connects the nodes of the Erlang translations of the
%% distribution tests, which utils/conformance/nodes.py runs with an EPMD of
%% its own. A node is named by its role, and uses the cookie in MERK_COOKIE,
%% as the test's nodes do. A node that listens says so first, with the name
%% the others connect to.

start_node(Role) ->
    {ok, _} = net_kernel:start([list_to_atom(Role), shortnames]),
    erlang:set_cookie(node(), list_to_atom(os:getenv("MERK_COOKIE"))).

listen_node(Role) ->
    start_node(Role),
    io:format("listening ~s~n", [node()]).

connect_node(Address) ->
    Node = list_to_atom(Address),
    pong = net_adm:ping(Node),
    Node.

%% The pid registered under a name on a node, as lookup finds a mailbox.
lookup_name(Node, Name) ->
    case rpc:call(Node, erlang, whereis, [Name]) of
        Pid when is_pid(Pid) -> {ok, Pid};
        _ -> error
    end.
