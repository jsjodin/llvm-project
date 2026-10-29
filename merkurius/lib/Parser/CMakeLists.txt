add_merkurius_library(MerkuriusParser
  char-block.cpp
  convert.cpp
  convert-goals.cpp
  convert-pragmas.cpp
  convert-directives.cpp
  lexer.cpp
  message.cpp
  operators.cpp
  parse-tree.cpp
  provenance.cpp
  source.cpp
  term-reader.cpp
  term.cpp
  token.cpp
  unparse.cpp

  LINK_LIBS
  MerkuriusCommon

  LINK_COMPONENTS
  Support
  )
