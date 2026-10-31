% Types for imported.m: a base type, an exported subtype, a subtype
% exported abstractly, and an abstract type.
:- module fruits.
:- interface.
:- type fruit ---> apple ; pear ; lemon ; orange.
:- type citrus =< fruit ---> lemon ; orange.
:- type pome.
:- type secret.
:- implementation.
:- type pome =< fruit ---> apple ; pear.
:- type secret ---> secret.
