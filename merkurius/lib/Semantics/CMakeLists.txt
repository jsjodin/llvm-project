add_merkurius_library(MerkuriusSemantics
  desugar.cpp
  module-loader.cpp
  quantify.cpp
  resolve-names.cpp
  semantic-form.cpp
  symbols.cpp

  LINK_LIBS
  MerkuriusCommon
  MerkuriusParser

  LINK_COMPONENTS
  Support
  )
