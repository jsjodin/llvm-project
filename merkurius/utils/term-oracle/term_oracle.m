% The term oracle reads the items on its standard input with the Mercury
% standard library's term parser, and prints each item's term in the
% canonical notation of merkc -fdebug-dump-terms (lib/Parser/term.cpp), one
% per line, or "error" for an item that does not parse. Comparing the two
% outputs tests our term reader against Mercury's. This program only uses
% the library through its documented interface.

:- module term_oracle.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module char, float, int, integer, list, mercury_term_parser.
:- import_module string, term, term_io, varset.

main(!IO) :-
    mercury_term_parser.read_term(Result, !IO),
    (
        Result = eof
    ;
        Result = error(_, _),
        io.write_string("error\n", !IO),
        main(!IO)
    ;
        Result = term(VarSet, Term),
        io.write_string(canonical(VarSet, Term) ++ "\n", !IO),
        main(!IO)
    ).

:- func canonical(varset(generic), term(generic)) = string.

canonical(VarSet, variable(Var, _)) =
    ( if varset.search_name(VarSet, Var, Name) then Name else "_" ).
canonical(VarSet, functor(Const, Args, _)) = Text :-
    (
        Const = atom(Name),
        Head = name(Name, Args)
    ;
        Const = integer(_, Int, Signedness, Size),
        Head = integer.to_string(Int) ++ integer_suffix(Signedness, Size)
    ;
        Const = string(String),
        Head = quoted(String, '"')
    ;
        Const = float(Float),
        Head = float_text(Float)
    ;
        Const = implementation_defined(Name),
        Head = "$" ++ Name
    ),
    (
        Args = [],
        Text = Head
    ;
        Args = [_ | _],
        ArgTexts = list.map(canonical(VarSet), Args),
        Text = Head ++ "(" ++ string.join_list(", ", ArgTexts) ++ ")"
    ).

:- func integer_suffix(signedness, integer_size) = string.

integer_suffix(Signedness, Size) = Suffix :-
    ( if Signedness = signed, Size = size_word then
        Suffix = ""
    else
        Letter = ( if Signedness = unsigned then "u" else "i" ),
        (
            Size = size_word,
            Bits = ""
        ;
            Size = size_8_bit,
            Bits = "8"
        ;
            Size = size_16_bit,
            Bits = "16"
        ;
            Size = size_32_bit,
            Bits = "32"
        ;
            Size = size_64_bit,
            Bits = "64"
        ),
        Suffix = Letter ++ Bits
    ).

    % [] and {} are unquoted only as constants.
    %
:- func name(string, list(term(generic))) = string.

name(Name, Args) =
    ( if
        ( alphanumeric_name(Name)
        ; Args = [], ( Name = "[]" ; Name = "{}" )
        )
    then
        Name
    else
        quoted(Name, '\'')
    ).

:- pred alphanumeric_name(string::in) is semidet.

alphanumeric_name(Name) :-
    string.index(Name, 0, First),
    char.is_lower(First),
    string.all_match(char.is_alnum_or_underscore, Name).

:- func quoted(string, char) = string.

quoted(Text, Quote) = String :-
    Q = string.from_char(Quote),
    Escaped = string.foldl(escape(Quote), Text, ""),
    String = Q ++ Escaped ++ Q.

:- func escape(char, char, string) = string.

escape(Quote, Char, Acc) = Acc ++ Escaped :-
    Code = char.to_int(Char),
    ( if Char = ('\\') then
        Escaped = "\\\\"
    else if Char = '\n' then
        Escaped = "\\n"
    else if Char = '\t' then
        Escaped = "\\t"
    else if Char = '\r' then
        Escaped = "\\r"
    else if Char = Quote then
        Escaped = "\\" ++ string.from_char(Char)
    else if ( Code < 0x20 ; Code = 0x7f ) then
        Escaped = string.format("\\x%x\\", [i(Code)])
    else
        Escaped = string.from_char(Char)
    ).

    % The shortest %g form that reads back as the same float, with a
    % fraction or an exponent, as PrintFloat in lib/Parser/term.cpp.
    %
:- func float_text(float) = string.

float_text(Float) = Text :-
    Shortest = shortest_g(Float, 1),
    ( if float.is_infinite(Float) then
        % There is no literal for infinity; this is one that reads as it.
        Text = ( if Float < 0.0 then "-1.0e+309" else "1.0e+309" )
    else if
        not string.contains_char(Shortest, '.'),
        not string.contains_char(Shortest, 'e'),
        not string.contains_char(Shortest, 'n')
    then
        Text = Shortest ++ ".0"
    else if
        string.sub_string_search(Shortest, "e", E),
        not string.contains_char(Shortest, '.')
    then
        string.split(Shortest, E, Mantissa, Exponent),
        Text = Mantissa ++ ".0" ++ Exponent
    else
        Text = Shortest
    ).

:- func shortest_g(float, int) = string.

shortest_g(Float, Precision) = Text :-
    G = string.format("%.*g", [i(Precision), f(Float)]),
    ( if
        ( Precision >= 17
        ; string.to_float(G, Back), Back = Float
        )
    then
        Text = G
    else
        Text = shortest_g(Float, Precision + 1)
    ).
