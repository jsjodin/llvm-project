add_merkurius_library(MerkuriusSemantics
  desugar.cpp
  module-loader.cpp
  quantify.cpp
  semantic-form.cpp

  LINK_LIBS
  MerkuriusCommon
  MerkuriusParser

  LINK_COMPONENTS
  Support
  )
