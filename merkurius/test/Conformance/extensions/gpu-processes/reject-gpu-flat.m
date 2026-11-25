% DESIGN: GPU processes / The device runtime
% A message that a GPU process sends or receives is flat: scalars,
% enumerations, mailboxes and non-recursive types of them, and as a
% constructor's last field only, a string or an array of scalars. A string
% elsewhere, a list, a recursive type or two arrays are not, and neither
% is a list or a closure that a GPU process's code makes.
% ORACLE: none
%
% RUN: not merkc -fc1 -fprocess-declarations -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_gpu_flat.
:- interface.
:- import_module io, process.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list.

:- type named ---> named(string, int).
:- type last_string ---> tagged(int, string).
:- type union_string ---> text(string) ; number(int).
:- type with_list ---> numbers(list(int)).
:- type tree ---> leaf ; node(tree, int, tree).
:- type block ---> block(int, array(float)).
:- type two_blocks ---> two(array(int), array(int)).

    % Rejected: a string that is not a constructor's last field.
:- process k_named receives named on gpu.
% CHECK-DAG: reject-gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type reject_gpu_flat.named: string has no fixed size, which only a whole message may have
k_named(_, !IO).

    % Accepted: a string or an array as the last field.
:- process k_last receives last_string on gpu.
k_last(_, !IO).
:- process k_union receives union_string on gpu.
k_union(_, !IO).
:- process k_block receives block on gpu.
k_block(_, !IO).

    % Rejected: a list, a recursive type, two arrays.
:- process k_list receives with_list on gpu.
% CHECK-DAG: reject-gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type reject_gpu_flat.with_list: list.list(int) is recursive
k_list(_, !IO).
:- process k_tree receives tree on gpu.
% CHECK-DAG: reject-gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type reject_gpu_flat.tree: reject_gpu_flat.tree is recursive
k_tree(_, !IO).
:- process k_two receives two_blocks on gpu.
% CHECK-DAG: reject-gpu-flat.m:[[@LINE-1]]:4: error: a message that code running on a GPU sends or receives must be flat, but this one is of type reject_gpu_flat.two_blocks: array.array(int) has no fixed size, which only a whole message may have
k_two(_, !IO).

    % Rejected: making a list or a closure in a GPU process's code.
:- process k_makes(mailbox(int)) receives int on gpu.
k_makes(Out, _, !IO) :-
    L = [1, 2, 3],
% CHECK-DAG: reject-gpu-flat.m:[[@LINE-1]]:9: error: code of a process kind that runs on a GPU must make only flat values, but this makes a value of type list.list(int): list.list(int) is recursive
    Add = (pred(X::in, Y::out) is det :- Y = X + 1),
% CHECK-DAG: reject-gpu-flat.m:[[@LINE-1]]:12: error: code of a process kind that runs on a GPU must make only flat values, but this makes Add of type pred(int, int): pred(int, int) is a higher-order type
    Add(list.length(L), N),
    send(Out, N, !IO).

main(!IO).
