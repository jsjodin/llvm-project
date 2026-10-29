add_merkurius_library(MerkuriusParser
  char-block.cpp
  lexer.cpp
  provenance.cpp
  source.cpp
  token.cpp

  LINK_LIBS
  MerkuriusCommon

  LINK_COMPONENTS
  Support
  )
