:- module sibling_parent.
:- interface.
:- include_module first, second.
:- implementation.
:- import_module sibling_parent.second.
