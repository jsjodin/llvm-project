add_merkurius_library(MerkuriusSemantics
  check-determinism.cpp
  check-modes.cpp
  check-purity.cpp
  check-types.cpp
  desugar.cpp
  insts.cpp
  module-loader.cpp
  quantify.cpp
  resolve-names.cpp
  semantic-form.cpp
  symbols.cpp
  types.cpp

  LINK_LIBS
  MerkuriusCommon
  MerkuriusParser

  LINK_COMPONENTS
  Support
  )
