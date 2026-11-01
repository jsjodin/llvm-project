% An inst that shares its parts is exponentially larger as a tree: X30's
% inst doubles 30 times, and the message of the mode error in the
% unifications of Y, whose argument Z is never bound, cuts it short.
:- module inst_growth.
:- interface.
:- pred q is det.
:- implementation.
q :-
    X0 = 1,
    X1 = {X0, X0},
    X2 = {X1, X1},
    X3 = {X2, X2},
    X4 = {X3, X3},
    X5 = {X4, X4},
    X6 = {X5, X5},
    X7 = {X6, X6},
    X8 = {X7, X7},
    X9 = {X8, X8},
    X10 = {X9, X9},
    X11 = {X10, X10},
    X12 = {X11, X11},
    X13 = {X12, X12},
    X14 = {X13, X13},
    X15 = {X14, X14},
    X16 = {X15, X15},
    X17 = {X16, X16},
    X18 = {X17, X17},
    X19 = {X18, X18},
    X20 = {X19, X19},
    X21 = {X20, X20},
    X22 = {X21, X21},
    X23 = {X22, X22},
    X24 = {X23, X23},
    X25 = {X24, X24},
    X26 = {X25, X25},
    X27 = {X26, X26},
    X28 = {X27, X27},
    X29 = {X28, X28},
    X30 = {X29, X29},
    Y = {X30, Z},
    Y = {X30, Z}.
