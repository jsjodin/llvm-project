add_merkurius_library(MerkuriusSemantics
  desugar.cpp
  semantic-form.cpp

  LINK_LIBS
  MerkuriusCommon
  MerkuriusParser

  LINK_COMPONENTS
  Support
  )
