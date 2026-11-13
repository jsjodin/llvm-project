% Code that runs on a GPU sends and receives only flat messages, and a GPU
% process makes only flat values (design document, "Messages and memory on
% the device"; docs/INPROGRESS.md, GPU processes without omp target, step
% 5a). A flat value's bytes say all of it: a scalar, an enumeration, a
% mailbox, a record or tagged union of flat fields that is not recursive, a
% tuple of them, and, as a whole value, a string or an array of scalars or
% enumerations, or a value of a type whose constructors' fields are flat
% and of a fixed size but for each one's last, which may be such a block. A GPU process may make none other, in its kind's code and in
% the procedures of the module that it calls; a target region may, and its
% messages are flat. A kind that runs on the host is not checked. Each
% error names the type and why it is not flat.
%
% RUN: not merkc -fc1 -fprocess-declarations -fopenmp -fdebug-dump-determinism %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module gpu_flat.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, univ.

:- type colour
    --->    red
    ;       green.

:- type pos
    --->    pos(int, float).

    % Flat: fields that are scalars, an enumeration, a mailbox, a record
    % and a tuple.
:- type good
    --->    good(colour, mailbox(int), pos, {int, character})
    ;       none.

:- type named
    --->    named(string, int).

:- type tree
    --->    leaf
    ;       node(tree, int, tree).

:- type boxed
    --->    some [T] boxed(T).

:- type with_closure
    --->    with_closure(pred(int)).

    % A fixed part and one block, the last field of each constructor.
:- type with_block
    --->    block(int, colour, array(int))
    ;       text(pos, string)
    ;       empty.

:- type two_blocks
    --->    two_blocks(array(int), array(int)).

:- type block_of_strings
    --->    block_of_strings(int, array(string)).

:- type block_inside
    --->    block_inside(int, with_block).

    % Flat messages, as whole values: no error.
:- process good_kind receives good on gpu.
good_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process string_kind receives string on gpu.
string_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process colours_kind receives array(colour) on gpu.
colours_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process block_kind(mailbox(with_block)) receives with_block on gpu.
block_kind(Out, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(block(N, C, A)) then
        send(Out, block(N + 1, C, A), !IO)
    else
        send(Out, text(pos(1, 2.0), "no block"), !IO)
    ).

    % Messages that are not flat.
:- process named_kind receives named on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.named: string has no fixed size, which only a whole message may have
named_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process two_blocks_kind receives two_blocks on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.two_blocks: array.array(int) has no fixed size, which only a whole message may have
two_blocks_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process block_of_strings_kind receives block_of_strings on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.block_of_strings: array.array(string) has no fixed size, which only a whole message may have
block_of_strings_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process block_inside_kind receives block_inside on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.block_inside: array.array(int) has no fixed size, which only a whole message may have
block_inside_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process tree_kind receives tree on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.tree: gpu_flat.tree is recursive
tree_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process boxed_kind receives boxed on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.boxed: gpu_flat.boxed has an existentially typed constructor
boxed_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process closure_kind receives with_closure on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.with_closure: pred(int) is a higher-order type
closure_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process strings_kind receives array(string) on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type array.array(string): array.array(string)'s elements are not scalars or enumerations
strings_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process univ_kind receives univ on gpu.
% CHECK: gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type univ.univ: univ.univ is abstract or primitive here, so its representation is unknown
univ_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

    % A message that a GPU process sends.
:- process sender(mailbox(list(int))) receives int on gpu.
sender(Out, Self, !IO) :-
    receive(Self, any, infinity, _, !IO),
    send(Out, [], !IO).
% CHECK: gpu-flat.m:[[@LINE-1]]:5: error: a message that code running on a GPU sends or receives must be flat, but this one is of type list.list(int): list.list(int) is recursive

    % Values that a GPU process makes: a list in its own code, a closure,
    % and a record with a string in a procedure of the module that it calls,
    % reported once though the kind calls it twice.
:- process maker receives int on gpu.
maker(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        L = [N],
        % CHECK: gpu-flat.m:[[@LINE-1]]:13: error: code of a process kind that runs on a GPU must make only flat values, but this makes a value of type list.list(int): list.list(int) is recursive
        Add = (pred(X::in, Y::out) is det :- Y = X + list.length(L)),
        % CHECK: gpu-flat.m:[[@LINE-1]]:16: error: code of a process kind that runs on a GPU must make only flat values, but this makes Add of type pred(int, int): pred(int, int) is a higher-order type
        Add(N, M),
        named_of(M, _),
        named_of(N, _)
    else
        true
    ).

:- pred named_of(int::in, named::out) is det.
named_of(N, named("n", N)).
% CHECK: gpu-flat.m:[[@LINE-1]]:13: error: code of a process kind that runs on a GPU must make only flat values, but this makes a value of type gpu_flat.named: string has no fixed size, which only a whole message may have

    % A kind that runs on the host makes and receives what it likes.
:- process host_kind receives tree.
host_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO),
    _ = [1, 2].

main(!IO) :-
    new_mailbox(Trees, !IO),
    new_mailbox(Ints, !IO),
    send(Ints, 1, !IO),
    %$omp target
    (
        % A region makes a list, which goes when it ends: no error.
        receive(Ints, any, after(0), R, !IO),
        L = ( if R = message(N) then [N] else [] ),
        send(Ints, list.length(L), !IO),
        send(Trees, leaf, !IO)
        % CHECK: gpu-flat.m:[[@LINE-1]]:9: error: a message that code running on a GPU sends or receives must be flat, but this one is of type gpu_flat.tree: gpu_flat.tree is recursive
    ).
