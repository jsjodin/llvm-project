% A process kind that runs on a GPU and takes a list, for
% process-kernel-errors.m.
:- module process_kernel_list.
:- interface.
:- import_module io, list, process.
:- process lister(list(int)) receives int on gpu.
:- implementation.

lister(_, _, !IO).
