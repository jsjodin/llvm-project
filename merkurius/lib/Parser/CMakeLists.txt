add_merkurius_library(MerkuriusParser
  char-block.cpp
  convert.cpp
  lexer.cpp
  message.cpp
  operators.cpp
  parse-tree.cpp
  provenance.cpp
  source.cpp
  term-reader.cpp
  term.cpp
  token.cpp

  LINK_LIBS
  MerkuriusCommon

  LINK_COMPONENTS
  Support
  )
